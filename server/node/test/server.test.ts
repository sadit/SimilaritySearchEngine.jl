// The client against a running server (PLAN.md §5.17, phase 10b). `api/21` of the server's full
// suite runs this file with SSE_TEST_URL, the root of the shared test server. Without it, every
// test is skipped. The routes themselves are tested by the server's own files; this file checks
// that the client sends what they accept and reads what they answer.

import { describe, test, before } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  HasFollowers,
  InvalidOption,
  InvalidRequest,
  JobFailed,
  NotFound,
  PluginNotLoaded,
  ResourceNotFound,
  SimilaritySearchClient,
  StateConflict,
  UnknownFollower,
  UnknownPartition,
  UnsupportedOperation,
  WrongDimension,
} from "../src/index.js";
import type { SearchHit } from "../src/index.js";

const url = process.env.SSE_TEST_URL;
const skip = url === undefined ? "SSE_TEST_URL is not set: these tests need a server" : false;

interface Row {
  doc_id: string;
  text: string;
  vector: number[];
  keywords: string[];
}

const N = 240;
let rows: Row[] = [];
/** Row `i`, counted from 1 as the `_id`s are: the items are appended in order. */
const row = (i: number): Row => rows[i - 1]!;
const vec = (i: number): Float32Array => new Float32Array(row(i).vector);
const repoOf = (i: number): string => (i % 4 === 0 ? "cuatro" : "otro");
const range = (a: number, b: number): number[] => Array.from({ length: b - a + 1 }, (_, j) => a + j);
const ids = (hits: SearchHit[]): number[] => hits.map((h) => h._id);
const POLL = { pollInterval: 500 };

/** Runs `f` and returns what it threw. Fails when it does not throw. */
async function thrown(f: () => Promise<unknown>): Promise<Error> {
  try {
    await f();
  } catch (e) {
    return e as Error;
  }
  assert.fail("no error was thrown");
}

describe("the client against a server", { skip }, () => {
  const encodings: (string | null)[] = [];
  let client: SimilaritySearchClient;

  before(() => {
    const data = process.env.SSE_TEST_DATA ?? fileURLToPath(new URL("../../../../test/data/frankenstein.jsonl", import.meta.url));
    rows = readFileSync(data, "utf8").split("\n").filter((l) => l.trim() !== "").slice(0, N).map((l) => JSON.parse(l) as Row);
    // A fetch that records the Content-Encoding of each response, to check the compression.
    const recording = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
      const response = await fetch(input, init);
      encodings.push(response.headers.get("content-encoding"));
      return response;
    };
    client = new SimilaritySearchClient(url, { fetch: recording as typeof globalThis.fetch });
  });

  test("control routes", async () => {
    assert.equal((await client.healthz()).status, "ok");
    assert.equal((await client.readyz()).status, "ok");
    assert.match(await client.metrics(), /simsearch_/);
  });

  test("a dense dataset", async () => {
    const created = await client.createDataset({
      id: "nc_dense", index_kind: "exhaustive_search", meta_schema: [{ name: "n", type: "int64" }], partition_by: "repo",
    });
    assert.deepEqual(created, { status: "created", id: "nc_dense" });
    const items = range(1, N).map((i) => ({ doc_id: row(i).doc_id, vector: vec(i), keywords: [`k${i % 3}`], n: i, repo: repoOf(i) }));
    assert.equal((await client.append("nc_dense", items)).inserted, N);

    const info = await client.getDataset("nc_dense");
    assert.equal(info.doc_count, N);
    assert.equal(info.loaded, true);
    assert.equal(info.max_k, 100);
    assert.equal(info.partition_by, "repo");
    assert.deepEqual(info.meta_schema, [{ name: "n", type: "int64" }]);
    assert.ok((await client.listDatasets()).datasets.some((d) => d.id === "nc_dense"));

    // An exact index returns the true nearest neighbors.
    const q = vec(9);
    const dist = (i: number) => vec(i).reduce((s, x, j) => s + (x - q[j]!) ** 2, 0);
    const want = range(1, N).sort((a, b) => dist(a) - dist(b)).slice(0, 5).map((i) => row(i).doc_id);
    const res = await client.search("nc_dense", q, { k: 5 });
    assert.equal(res.status, "searched");
    assert.deepEqual(res.results.map((r) => r.doc_id), want);
    assert.equal(res.results[0]!._id, 9);

    // A filter, a partition, and their errors.
    const filtered = await client.search("nc_dense", q, { k: 5, filter: { n: { gte: 200 } } });
    assert.ok(filtered.results.length > 0 && filtered.results.every((r) => r._id >= 200));
    const part = await client.search("nc_dense", q, { k: 5, partition: "cuatro" });
    assert.ok(part.results.length === 5 && part.results.every((r) => r._id % 4 === 0));
    const partitions = await client.getPartitions("nc_dense");
    assert.deepEqual(Object.fromEntries(partitions.partitions.map((p) => [p.name, p.items])), { cuatro: 60, otro: 180 });

    const e = await thrown(() => client.search("nc_dense", q, { k: 5, partition: "cuatr" }));
    assert.ok(e instanceof UnknownPartition && e instanceof NotFound);
    assert.equal(e.status, 404);
    assert.match(e.message, /cuatro/);
    assert.ok((await thrown(() => client.getDataset("nc_missing"))) instanceof ResourceNotFound);
    assert.ok((await thrown(() => client.search("nc_missing", q))) instanceof ResourceNotFound);
    assert.ok((await thrown(() => client.search("nc_dense", q, { beamsearch_overrides: { bsize: 8 } }))) instanceof InvalidRequest);
    // A query and an item of another dimension. Before 2026-11-24 the engine did not check the
    // dimension of a dense dataset of Float32 vectors, and the server answered 500.
    const dim = await thrown(() => client.search("nc_dense", [1, 2, 3]));
    assert.ok(dim instanceof WrongDimension && dim instanceof InvalidRequest);
    assert.equal(dim.status, 400);
    assert.equal(dim.kind, "WrongDimension");
    const item = await thrown(() => client.append("nc_dense", [{ doc_id: "nc_short", vector: [1, 2, 3] }]));
    assert.ok(item instanceof WrongDimension);

    // fetch and exists take a `_id` or a `doc_id`.
    const fetched = await client.fetch("nc_dense", [5, row(6).doc_id, "no-such-id"]);
    assert.deepEqual(fetched.results.map((r) => r._id), [5, 6]);
    const first = fetched.results[0]!;
    assert.equal(first.doc_id, row(5).doc_id);
    assert.deepEqual(first.keywords, ["k2"]);
    assert.equal(first.n, 5);
    assert.equal(first.repo, "otro");
    const ex = await client.exists("nc_dense", [5, "no-such-id"]);
    assert.deepEqual(ex.results.map((r) => [r.id, r._id, r.exists, r.deleted]), [["5", 5, true, false], ["no-such-id", null, false, false]]);

    assert.deepEqual(await client.deleteItem("nc_dense", 9), { status: "soft_deleted", id: "nc_dense", _id: 9 });
    assert.equal((await client.exists("nc_dense", [9])).results[0]!.deleted, true);
    assert.ok(!ids((await client.search("nc_dense", q, { k: 5 })).results).includes(9));

    const log = await client.getLog("nc_dense", { limit: 1 });
    assert.equal(log.entries.length, 1);
    assert.ok(log.total > 1);
  });

  test("a search answers up to max_k results, compressed", async () => {
    const q = vec(21);
    // Item 9 is deleted, and the server drops deleted results.
    assert.equal((await client.search("nc_dense", q, { k: 50 })).results.length, 49);
    encodings.length = 0;
    assert.equal((await client.search("nc_dense", q, { k: 100 })).results.length, 99);
    // The fetch of Node.js asks for gzip and decompresses the response.
    assert.deepEqual(encodings, ["gzip"]);
    const e = await thrown(() => client.search("nc_dense", q, { k: 101 }));
    assert.ok(e instanceof InvalidOption);
    assert.match(e.message, /max_k = 100/);

    // unload and reload, with a new max_k.
    assert.deepEqual(await client.unloadDataset("nc_dense"), { status: "unloaded", id: "nc_dense" });
    assert.ok((await thrown(() => client.search("nc_dense", q))) instanceof ResourceNotFound);
    assert.deepEqual(await client.reloadDataset("nc_dense", { max_k: 200 }), { status: "reloaded", id: "nc_dense" });
    assert.equal((await client.getDataset("nc_dense")).max_k, 200);
    assert.equal((await client.search("nc_dense", q, { k: 150 })).results.length, 149);
  });

  test("a graph and its calibration", async () => {
    await client.createDataset({ id: "nc_graph", index_kind: "searchgraph", distance: "L2" });
    await client.append("nc_graph", range(1, N).map((i) => ({ doc_id: row(i).doc_id, vector: row(i).vector })));
    const cal = await client.calibrate("nc_graph", { minrecall: 0.9, numqueries: 16 });
    assert.equal(cal.status, "calibrated");
    // The server keeps the levels as Float32: 0.9 arrives as 0.8999999761581421.
    assert.ok(Math.abs(cal.calibration.minrecall - 0.9) < 1e-6);
    assert.ok(cal.calibration.recall >= 0 && cal.calibration.recall <= 1);
    assert.equal((await client.getDataset("nc_graph")).beamsearch_baseline?.bsize, cal.baseline.bsize);
    const res = await client.search("nc_graph", vec(12), { k: 3, beamsearch_overrides: { bsize: cal.baseline.bsize } });
    assert.equal(res.results[0]!._id, 12);
    assert.ok((await thrown(() => client.calibrate("nc_graph", { partition: "otro" }))) instanceof InvalidOption);
  });

  test("a text dataset, a hybrid search and several fields", async () => {
    await client.createDataset({ id: "nc_text", index_kind: "bm25_invfile" });
    const appended = await client.append("nc_text", range(1, N).map((i) => ({ doc_id: row(i).doc_id, text: row(i).text })));
    assert.equal(appended.inserted, N);
    const res = await client.ftsearch("nc_text", row(30).text, { k: 5 });
    assert.equal(res.results.length, 5);
    assert.equal(res.results[0]!.doc_id, row(30).doc_id);
    assert.ok((await thrown(() => client.ftsearch("nc_text", "monster", { partition: "otro" }))) instanceof InvalidOption);

    const report = await client.getVocab("nc_text");
    assert.equal(report.id, "nc_text");
    assert.ok(report.vocsize > 0 && report.queries >= 1);
    assert.equal(typeof (await client.getVocab("nc_text", { scan: true })).live_sampled, "boolean");
    const postings = await client.postingList("nc_text", "the", { limit: 3 });
    assert.ok(postings.ids.length <= 3 && postings.ids.every((x, j) => j === 0 || postings.ids[j - 1]! < x));

    const fused = await client.hybridSearch("nc_graph", "nc_text", { vector: vec(30), text: row(30).text, k: 4 });
    assert.equal(fused.results.length, 4);
    assert.equal(fused.results[0]!._id, 30);
    assert.equal(fused.results[0]!.doc_id, row(30).doc_id);
    assert.ok(fused.results.every((r, j) => j === 0 || fused.results[j - 1]!.score >= r.score));

    const fields = await client.ftsearchFields({ a: "nc_text", b: "nc_text" }, row(30).text, { k: 3 });
    assert.deepEqual(Object.keys(fields), ["a", "b"]);
    assert.deepEqual(fields.a, fields.b);
    assert.equal(fields.a!.length, 3);
  });

  test("followers", async () => {
    const textItem = (i: number) => ({ doc_id: row(i).doc_id, text: row(i).text, keywords: row(i).keywords, repo: repoOf(i) });
    await client.createDataset({ id: "nc_fl", index_kind: "bm25_invfile", max_k: 120 });
    await client.append("nc_fl", range(1, 120).map(textItem));
    const f = await client.createFollower("nc_fl", { name: "repo", kind: "partition", source: "meta:repo" });
    assert.equal(f.status, "created");
    assert.equal(f.name, "nc_fl/repo");
    assert.equal(f.kind, "PartitionFollower");
    assert.equal(f.synchronized, true);
    assert.equal(f.stats?.partitions, 2);
    await client.createFollower("nc_fl", { name: "kw", kind: "keyword", source: "keywords", policy: "manual" });
    assert.deepEqual((await client.listFollowers("nc_fl")).followers.map((s) => s.name).sort(), ["nc_fl/kw", "nc_fl/repo"]);
    const parts = await client.followerPartitions("nc_fl", "repo");
    assert.deepEqual(Object.fromEntries(parts.partitions.map((p) => [p.name, p.items])), { cuatro: 30, otro: 90 });

    // A partition is the global search restricted to its items.
    const q = row(8).text;
    const want = ids((await client.ftsearch("nc_fl", q, { k: 120 })).results).filter((x) => x % 4 === 0).slice(0, 5);
    assert.deepEqual(ids((await client.followerFtsearch("nc_fl", "repo=cuatro", q, { k: 5 })).results), want);
    assert.ok((await thrown(() => client.followerFtsearch("nc_fl", "repo=cinco", q, { k: 3 }))) instanceof UnknownPartition);
    assert.ok((await thrown(() => client.getFollower("nc_fl", "rep"))) instanceof UnknownFollower);

    // A keyword follower: a search by keys and a posting list.
    const key = "Elizabeth";
    const withKey = range(1, 120).filter((i) => row(i).keywords.includes(key));
    assert.ok(withKey.length > 2);
    assert.deepEqual((await client.postingList("nc_fl", key, { follower: "kw" })).ids, withKey);
    assert.deepEqual((await client.postingList("nc_fl", key, { follower: "kw", limit: 2 })).ids, withKey.slice(0, 2));
    const byKeys = await client.followerSearch("nc_fl", "kw", { keys: [key], k: 50 });
    assert.deepEqual(ids(byKeys.results).sort((a, b) => a - b), withKey);
    assert.ok((await thrown(() => client.followerSearch("nc_fl", "repo", { keys: [key], k: 5 }))) instanceof UnsupportedOperation);

    // `append` processes the on_index follower; the manual one lags until it is synchronized.
    await client.append("nc_fl", range(121, 140).map(textItem));
    assert.equal((await client.getFollower("nc_fl", "repo")).position, 140);
    assert.equal((await client.getFollower("nc_fl", "kw")).lag, 20);
    const synced = await client.syncFollower("nc_fl", "kw");
    assert.equal(synced.processed, 20);
    assert.equal(synced.synchronized, true);
    const rebuilt = await client.rebuildFollower("nc_fl", "kw");
    assert.equal(rebuilt.status, "rebuilt");
    assert.equal(rebuilt.position, 140);
    const group = await client.getGroup("nc_fl");
    assert.equal(group.leader.id, "nc_fl");
    assert.equal(group.followers.length, 2);

    // A dataset with followers is not deleted.
    assert.ok((await thrown(() => client.deleteDataset("nc_fl"))) instanceof HasFollowers);
    assert.deepEqual(await client.deleteFollower("nc_fl", "kw"), { status: "deleted", id: "nc_fl", follower: "kw" });
    await client.deleteFollower("nc_fl", "repo");
    assert.deepEqual((await client.listFollowers("nc_fl")).followers, []);
    await client.deleteDataset("nc_fl");
  });

  test("the partitions of a dense dataset, moved to a follower", async () => {
    await client.createDataset({ id: "nc_fl_dense", index_kind: "searchgraph", partition_by: "repo", partition_graph_min: 100 });
    await client.append("nc_fl_dense", range(1, N).map((i) => ({ doc_id: row(i).doc_id, vector: vec(i), repo: repoOf(i) })));
    const migrated = await client.migratePartitions("nc_fl_dense", { name: "repo", disjoint: true });
    assert.equal(migrated.status, "migrated");
    const parts = await client.followerPartitions("nc_fl_dense", "repo");
    assert.deepEqual(Object.fromEntries(parts.partitions.map((p) => [p.name, p.graph])), { cuatro: false, otro: true });
    assert.equal((await client.getDataset("nc_fl_dense")).partition_by, null);
    const res = await client.followerSearch("nc_fl_dense", "repo=otro", { vector: vec(13), k: 5 });
    assert.equal(res.results.length, 5);
    assert.equal(res.results[0]!._id, 13);
    assert.ok(res.results.every((r) => r._id % 4 !== 0));
    const cal = await client.followerCalibrate("nc_fl_dense", "repo=otro", { minrecall: 0.9, numqueries: 16 });
    assert.equal(cal.status, "calibrated");
    assert.equal(cal.follower, "repo=otro");
    await client.deleteFollower("nc_fl_dense", "repo");
    await client.deleteDataset("nc_fl_dense");
  });

  test("a key source of a plugin of the server", async () => {
    const listed = (await client.listPlugins()).plugins.find((p) => p.name === "SSETestPlugin");
    assert.deepEqual(listed?.key_sources, ["SSETestPlugin.decade"]);
    await client.createDataset({ id: "nc_plugin", index_kind: "bm25_invfile" });
    await client.append("nc_plugin", range(1, 40).map((i) => ({ doc_id: row(i).doc_id, text: row(i).text, year: 1960 + i })));
    const f = await client.createFollower("nc_plugin", { name: "decade", kind: "partition", source: { name: "SSETestPlugin.decade", field: "year" } });
    assert.equal(f.plugin, "SSETestPlugin");
    assert.equal(f.stats?.partitions, 5);
    const e = await thrown(() => client.createFollower("nc_plugin", { name: "x", kind: "partition", source: "Nope.decade" }));
    assert.ok(e instanceof PluginNotLoaded);
    await client.deleteFollower("nc_plugin", "decade");
    await client.deleteDataset("nc_plugin");
  });

  test("jobs", async () => {
    // Each job is a process on the server, so the three run at the same time.
    const [knn, missing, onText] = await Promise.allSettled([
      client.runJob("allknn", { dataset: "nc_dense", k: 3 }, POLL),
      client.runJob("allknn", { dataset: "nc_missing" }, POLL),
      client.runJob("fft", { dataset: "nc_text", k: 3 }, POLL),
    ]);
    assert.equal(knn.status, "fulfilled");
    const record = (knn as PromiseFulfilledResult<Awaited<ReturnType<typeof client.runJob>>>).value;
    assert.equal(record.kind, "allknn");
    const lines = (await client.getJobResult(record.id)).trim().split("\n").map((l) => JSON.parse(l) as { neighbors: number[]; dists: number[] });
    assert.equal(lines.length, N);
    assert.ok(lines.every((l) => l.neighbors.length === 3 && l.dists.length === 3));

    // A failed job carries the category of the engine error, read from its exit code.
    assert.equal(missing.status, "rejected");
    const e1 = (missing as PromiseRejectedResult).reason as JobFailed;
    assert.ok(e1 instanceof JobFailed);
    assert.equal(e1.exitCode, 1);
    assert.equal(e1.category, null);
    assert.equal(onText.status, "rejected");
    const e2 = (onText as PromiseRejectedResult).reason as JobFailed;
    assert.ok(e2 instanceof JobFailed);
    assert.equal(e2.jobKind, "fft");
    assert.equal(e2.exitCode, 2);
    assert.equal(e2.category, InvalidRequest);

    // The routes of one job.
    const info = await client.getJob(record.id);
    assert.equal(info.status, "completed");
    assert.ok((await client.listJobs({ kind: "allknn" })).jobs.some((j) => j.id === record.id && j.status === "completed"));
    assert.ok((await thrown(() => client.cancelJob(record.id))) instanceof StateConflict);
    assert.ok((await thrown(() => client.killJob(record.id))) instanceof StateConflict);
    assert.ok((await thrown(() => client.resumeJob(record.id))) instanceof StateConflict);
    assert.ok((await thrown(() => client.getJob("no-such-job"))) instanceof ResourceNotFound);

    // dump writes a bundle on the server, and load creates a dataset from it.
    const dumped = await client.runJob("dump", { dataset: "nc_graph" }, POLL);
    const bundle = JSON.parse(await client.getJobResult(dumped.id)) as { bundle_dir: string; manifest: Record<string, unknown> };
    assert.equal(typeof bundle.bundle_dir, "string");
    assert.ok(Object.keys(bundle.manifest).length > 0);
    // load writes the dataset to the working directory of the server, and reload opens it there.
    await client.runJob("load", { bundle: bundle.bundle_dir, dataset: "nc_graph_copy" }, POLL);
    assert.equal((await client.getDataset("nc_graph_copy")).loaded, false);
    assert.deepEqual(await client.reloadDataset("nc_graph_copy"), { status: "reloaded", id: "nc_graph_copy" });
    assert.equal((await client.getDataset("nc_graph_copy")).doc_count, N);
    assert.equal((await client.search("nc_graph_copy", vec(12), { k: 1 })).results[0]!.doc_id, row(12).doc_id);
  });

  test("administration", async () => {
    const created = await client.createToken({ user: "nc", permissions: ["read:nc_text"] });
    assert.equal(created.status, "created");
    assert.deepEqual(created.permissions, ["read:nc_text"]);
    const listed = (await client.listTokens()).tokens.find((t) => t.token === created.token);
    assert.equal(listed?.user, "nc");
    assert.deepEqual(await client.revokeToken(created.token), { status: "revoked" });
    assert.ok(!(await client.listTokens()).tokens.some((t) => t.token === created.token));
    assert.equal(typeof (await client.pruneTokens()).pruned, "number");
    assert.ok(Array.isArray((await client.jobsGc({ retention_seconds: 86400 })).jobs_purged));
  });

  test("the datasets of this file are deleted", async () => {
    for (const id of ["nc_dense", "nc_graph", "nc_text", "nc_graph_copy"]) {
      assert.deepEqual(await client.deleteDataset(id), { status: "deleted", id });
    }
    const left = (await client.listDatasets({ limit: 1000 })).datasets.filter((d) => d.id.startsWith("nc_"));
    assert.deepEqual(left, []);
  });
});
