// The tests of the client without a server (PLAN.md §5.17): a `fetch` given to the constructor
// records each request and answers it. The tests against a server are in server.test.ts.

import { describe, test } from "node:test";
import assert from "node:assert/strict";

import {
  ConflictingState,
  CorruptedStorage,
  EngineError,
  ERROR_KINDS,
  Forbidden,
  InvalidRequest,
  JobFailed,
  JobTimeout,
  NotFound,
  RemoteError,
  RequestRejected,
  ResourceNotFound,
  ServerFailure,
  SimilaritySearchClient,
  StateConflict,
  StorageFailure,
  Unauthenticated,
  UnexpectedResponse,
  UnknownPartition,
  WrongDimension,
  PendingBacklog,
} from "../src/index.js";

interface Call {
  method: string;
  url: URL;
  headers: Record<string, string>;
  body: unknown;
  signal: AbortSignal | null;
}

interface Answer {
  status?: number;
  /** An object is sent as JSON, and a string as it is. */
  body?: unknown;
}

/** A client whose `fetch` records every request and answers with `answer(call)`. */
function mockClient(answer: (call: Call) => Answer = () => ({ body: { status: "ok" } }), options: { token?: string; timeout?: number; url?: string } = {}) {
  const calls: Call[] = [];
  const fetch = async (input: string | URL | Request, init: RequestInit = {}): Promise<Response> => {
    const raw = init.body;
    const call: Call = {
      method: init.method ?? "GET",
      url: new URL(String(input)),
      headers: Object.fromEntries(Object.entries((init.headers ?? {}) as Record<string, string>)),
      body: raw === undefined ? undefined : JSON.parse(String(raw)),
      signal: init.signal ?? null,
    };
    calls.push(call);
    const { status = 200, body = { status: "ok" } } = answer(call);
    return new Response(typeof body === "string" ? body : JSON.stringify(body), { status });
  };
  const { url = "http://127.0.0.1:8080", ...rest } = options;
  const client = new SimilaritySearchClient(url, { ...rest, fetch: fetch as typeof globalThis.fetch });
  return { client, calls };
}

describe("requests", () => {
  // Each route: the call, the method, the path with its query, and the body.
  const routes: [string, (c: SimilaritySearchClient) => Promise<unknown>, string, string, unknown?][] = [
    ["healthz", (c) => c.healthz(), "GET", "/healthz"],
    ["readyz", (c) => c.readyz(), "GET", "/readyz"],
    ["createDataset", (c) => c.createDataset({ id: "ds", index_kind: "sparse_invfile", dimension: 30, max_k: 500 }),
      "POST", "/api/v1/datasets", { id: "ds", index_kind: "sparse_invfile", dimension: 30, max_k: 500 }],
    ["createDataset without options", (c) => c.createDataset(), "POST", "/api/v1/datasets", {}],
    ["listDatasets", (c) => c.listDatasets({ offset: 2, limit: 5 }), "GET", "/api/v1/datasets?offset=2&limit=5"],
    ["listDatasets without options", (c) => c.listDatasets(), "GET", "/api/v1/datasets"],
    ["getDataset", (c) => c.getDataset("ds"), "GET", "/api/v1/datasets/ds"],
    ["deleteDataset", (c) => c.deleteDataset("ds"), "DELETE", "/api/v1/datasets/ds"],
    ["getLog", (c) => c.getLog("ds", { limit: 1 }), "GET", "/api/v1/datasets/ds/log?limit=1"],
    ["getVocab", (c) => c.getVocab("ds", { scan: true, sample: 0 }), "GET", "/api/v1/datasets/ds/vocab?scan=true&sample=0"],
    ["getPartitions", (c) => c.getPartitions("ds"), "GET", "/api/v1/datasets/ds/partitions"],
    ["append", (c) => c.append("ds", [{ doc_id: "a", text: "el gato", year: 2001 }]),
      "POST", "/api/v1/datasets/ds/append", { items: [{ doc_id: "a", text: "el gato", year: 2001 }] }],
    ["search", (c) => c.search("ds", [1, 2], { k: 3, filter: { year: { gte: 2000 } }, partition: "p" }),
      "POST", "/api/v1/datasets/ds/search", { k: 3, filter: { year: { gte: 2000 } }, partition: "p", vector: [1, 2] }],
    ["search with overrides", (c) => c.search("ds", [1], { beamsearch_overrides: { bsize: 8 } }),
      "POST", "/api/v1/datasets/ds/search", { beamsearch_overrides: { bsize: 8 }, vector: [1] }],
    ["ftsearch", (c) => c.ftsearch("ds", "gato", { k: 2 }), "POST", "/api/v1/datasets/ds/ftsearch", { k: 2, text: "gato" }],
    ["hybridSearch", (c) => c.hybridSearch("d", "t", { vector: [1, 2], text: "gato", k: 4 }),
      "POST", "/api/v1/search/hybrid", { dense_index: "d", lexical_index: "t", text: "gato", k: 4, vector: [1, 2] }],
    ["hybridSearch with text only", (c) => c.hybridSearch("d", "t", { text: "gato" }),
      "POST", "/api/v1/search/hybrid", { dense_index: "d", lexical_index: "t", text: "gato" }],
    ["fetch", (c) => c.fetch("ds", ["a", 2]), "POST", "/api/v1/datasets/ds/fetch", { ids: ["a", 2] }],
    ["exists", (c) => c.exists("ds", ["a", 2]), "GET", "/api/v1/datasets/ds/exists?ids=a%2C2"],
    ["deleteItem", (c) => c.deleteItem("ds", 7), "POST", "/api/v1/datasets/ds/delete", { _id: 7 }],
    ["calibrate", (c) => c.calibrate("ds", { minrecall: 0.9, numqueries: 32, partition: "p" }),
      "POST", "/api/v1/datasets/ds/calibrate", { minrecall: 0.9, numqueries: 32, partition: "p" }],
    ["postingList", (c) => c.postingList("ds", "gato", { limit: 10 }), "GET", "/api/v1/datasets/ds/postings?key=gato&limit=10"],
    ["postingList of a sparse dimension", (c) => c.postingList("ds", 7), "GET", "/api/v1/datasets/ds/postings?key=7"],
    ["postingList of a follower", (c) => c.postingList("ds", "gato", { follower: "kw" }),
      "GET", "/api/v1/datasets/ds/followers/kw/postings?key=gato"],
    ["createFollower", (c) => c.createFollower("ds", { name: "kw", kind: "keyword", source: "keywords", policy: { periodic: { seconds: 60 } } }),
      "POST", "/api/v1/datasets/ds/followers", { name: "kw", kind: "keyword", source: "keywords", policy: { periodic: { seconds: 60 } } }],
    ["listFollowers", (c) => c.listFollowers("ds"), "GET", "/api/v1/datasets/ds/followers"],
    ["getFollower", (c) => c.getFollower("ds", "kw"), "GET", "/api/v1/datasets/ds/followers/kw"],
    ["deleteFollower", (c) => c.deleteFollower("ds", "kw"), "DELETE", "/api/v1/datasets/ds/followers/kw"],
    ["syncFollower", (c) => c.syncFollower("ds", "kw"), "POST", "/api/v1/datasets/ds/followers/kw/sync"],
    ["rebuildFollower", (c) => c.rebuildFollower("ds", "kw"), "POST", "/api/v1/datasets/ds/followers/kw/rebuild"],
    ["followerPartitions", (c) => c.followerPartitions("ds", "sec"), "GET", "/api/v1/datasets/ds/followers/sec/partitions"],
    ["followerSearch with a vector", (c) => c.followerSearch("ds", "sec=calle", { vector: [1, 2], k: 2, minrecall: 0.9 }),
      "POST", "/api/v1/datasets/ds/followers/sec%3Dcalle/search", { k: 2, minrecall: 0.9, vector: [1, 2] }],
    ["followerSearch with keys", (c) => c.followerSearch("ds", "kw", { keys: ["gato", "perro"], t: 2 }),
      "POST", "/api/v1/datasets/ds/followers/kw/search", { keys: ["gato", "perro"], t: 2 }],
    ["followerFtsearch", (c) => c.followerFtsearch("ds", "sec=casa azul", "gato", { k: 5 }),
      "POST", "/api/v1/datasets/ds/followers/sec%3Dcasa%20azul/ftsearch", { k: 5, text: "gato" }],
    ["followerCalibrate", (c) => c.followerCalibrate("ds", "sec=calle", { minrecall: 0.95 }),
      "POST", "/api/v1/datasets/ds/followers/sec%3Dcalle/calibrate", { minrecall: 0.95 }],
    ["migratePartitions", (c) => c.migratePartitions("ds", { name: "by_repo", disjoint: false }),
      "POST", "/api/v1/datasets/ds/migrate-partitions", { name: "by_repo", disjoint: false }],
    ["getGroup", (c) => c.getGroup("ds"), "GET", "/api/v1/datasets/ds/group"],
    ["listPlugins", (c) => c.listPlugins(), "GET", "/api/v1/plugins"],
    ["submitJob", (c) => c.submitJob("allknn", { dataset: "ds", k: 8 }), "POST", "/api/v1/jobs/allknn", { dataset: "ds", k: 8 }],
    ["submitJob with a command", (c) => c.submitJob("searchbatch", { command: ["searchbatch", "--dataset", "ds"] }),
      "POST", "/api/v1/jobs/searchbatch", { command: ["searchbatch", "--dataset", "ds"] }],
    ["getJob", (c) => c.getJob("j1"), "GET", "/api/v1/jobs/j1"],
    ["listJobs", (c) => c.listJobs({ status: "failed", kind: "fft" }), "GET", "/api/v1/jobs?status=failed&kind=fft"],
    ["blockJob", (c) => c.blockJob("j1"), "POST", "/api/v1/jobs/j1/block"],
    ["resumeJob", (c) => c.resumeJob("j1"), "POST", "/api/v1/jobs/j1/resume"],
    ["killJob", (c) => c.killJob("j1"), "POST", "/api/v1/jobs/j1/kill"],
    ["cancelJob", (c) => c.cancelJob("j1"), "DELETE", "/api/v1/jobs/j1"],
    ["createToken", (c) => c.createToken({ user: "ana", permissions: ["read:*"] }),
      "POST", "/api/v1/admin/tokens", { user: "ana", permissions: ["read:*"] }],
    ["createToken without options", (c) => c.createToken(), "POST", "/api/v1/admin/tokens", { user: "anonymous", permissions: [] }],
    ["listTokens", (c) => c.listTokens(), "GET", "/api/v1/admin/tokens"],
    ["pruneTokens", (c) => c.pruneTokens(), "POST", "/api/v1/admin/tokens/prune"],
    ["revokeToken", (c) => c.revokeToken("abc"), "DELETE", "/api/v1/admin/tokens/abc"],
    ["jobsGc", (c) => c.jobsGc({ retention_seconds: 60 }), "POST", "/api/v1/admin/jobs/gc", { retention_seconds: 60 }],
    ["unloadDataset", (c) => c.unloadDataset("ds"), "POST", "/api/v1/admin/datasets/ds/unload"],
    ["reloadDataset", (c) => c.reloadDataset("ds"), "POST", "/api/v1/admin/datasets/ds/reload"],
    ["reloadDataset with max_k", (c) => c.reloadDataset("ds", { max_k: 1000 }),
      "POST", "/api/v1/admin/datasets/ds/reload", { max_k: 1000 }],
  ];

  for (const [name, invoke, method, path, body] of routes) {
    test(name, async () => {
      const { client, calls } = mockClient();
      await invoke(client);
      assert.equal(calls.length, 1);
      const call = calls[0]!;
      assert.equal(call.method, method);
      assert.equal(call.url.pathname + call.url.search, path);
      assert.deepEqual(call.body, body);
    });
  }

  test("every route of the server has a method, and the table tests each one", () => {
    // The server has 45 routes under /api/v1 and 3 of control. postingList serves two of them,
    // the posting list of a dataset and that of a follower.
    const ROUTE_METHODS = [
      "healthz", "readyz", "metrics",
      "createDataset", "listDatasets", "getDataset", "deleteDataset", "getLog", "getVocab", "getPartitions",
      "append", "search", "ftsearch", "hybridSearch", "fetch", "exists", "deleteItem", "calibrate", "postingList",
      "createFollower", "listFollowers", "getFollower", "deleteFollower", "syncFollower", "rebuildFollower",
      "followerPartitions", "followerSearch", "followerFtsearch", "followerCalibrate", "migratePartitions",
      "getGroup", "listPlugins",
      "submitJob", "getJob", "getJobResult", "listJobs", "blockJob", "resumeJob", "killJob", "cancelJob",
      "createToken", "listTokens", "pruneTokens", "revokeToken", "jobsGc", "unloadDataset", "reloadDataset",
    ];
    assert.equal(ROUTE_METHODS.length + 1, 48);
    const helpers = ["constructor", "request", "waitJob", "runJob", "ftsearchFields"];
    const own = Object.getOwnPropertyNames(SimilaritySearchClient.prototype).filter((n) => !helpers.includes(n));
    assert.deepEqual([...own].sort(), [...ROUTE_METHODS].sort());
    const tested = new Set(routes.map(([name]) => name.split(" ")[0]));
    const text = ["metrics", "getJobResult"];
    assert.deepEqual(ROUTE_METHODS.filter((m) => !tested.has(m) && !text.includes(m)), []);
  });

  test("a dataset id and a follower name are escaped as one path segment", async () => {
    const { client, calls } = mockClient();
    await client.getFollower("a b", "x/y=z");
    assert.equal(calls[0]!.url.pathname, "/api/v1/datasets/a%20b/followers/x%2Fy%3Dz");
  });

  test("a Float32Array is sent as a JSON array", async () => {
    const { client, calls } = mockClient();
    const v = new Float32Array([0.5, 0.25]);
    await client.append("ds", [{ doc_id: "a", vector: v }, { indices: new Int32Array([1, 3]), values: v }]);
    await client.search("ds", v);
    await client.calibrate("ds", { queries: [v] });
    await client.followerSearch("ds", "p=x", { vector: v });
    await client.hybridSearch("d", "t", { vector: v });
    assert.deepEqual(calls[0]!.body, { items: [{ doc_id: "a", vector: [0.5, 0.25] }, { indices: [1, 3], values: [0.5, 0.25] }] });
    assert.deepEqual((calls[1]!.body as { vector: number[] }).vector, [0.5, 0.25]);
    assert.deepEqual((calls[2]!.body as { queries: number[][] }).queries, [[0.5, 0.25]]);
    assert.deepEqual((calls[3]!.body as { vector: number[] }).vector, [0.5, 0.25]);
    assert.deepEqual((calls[4]!.body as { vector: number[] }).vector, [0.5, 0.25]);
  });

  test("the token, the content type and the root of the server", async () => {
    const { client, calls } = mockClient(undefined, { token: "t0k", url: "http://example.org:9/" });
    await client.search("ds", [1]);
    await client.getDataset("ds");
    assert.equal(client.url, "http://example.org:9");
    assert.equal(calls[0]!.url.origin, "http://example.org:9");
    assert.equal(calls[0]!.headers.authorization, "Bearer t0k");
    assert.equal(calls[0]!.headers["content-type"], "application/json");
    assert.equal(calls[1]!.headers["content-type"], undefined);
    const anonymous = mockClient();
    await anonymous.client.healthz();
    assert.equal(anonymous.calls[0]!.headers.authorization, undefined);
  });

  test("the timeout is a signal of the request, and 0 sends none", async () => {
    const timed = mockClient();
    await timed.client.healthz();
    assert.ok(timed.calls[0]!.signal instanceof AbortSignal);
    const untimed = mockClient(undefined, { timeout: 0 });
    await untimed.client.healthz();
    assert.equal(untimed.calls[0]!.signal, null);
  });

  test("a request that takes longer than the timeout fails with the error of the signal", async () => {
    const fetch = (_input: string | URL | Request, init: RequestInit = {}) =>
      new Promise<Response>((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => reject(init.signal?.reason));
      });
    const client = new SimilaritySearchClient("http://127.0.0.1:8080", { timeout: 20, fetch: fetch as typeof globalThis.fetch });
    await assert.rejects(client.healthz(), { name: "TimeoutError" });
  });

  test("metrics and the result of a job are text", async () => {
    const { client } = mockClient((call) => ({ body: call.url.pathname === "/metrics" ? "simsearch_up 1\n" : '{"id":1}\n{"id":2}\n' }));
    assert.equal(await client.metrics(), "simsearch_up 1\n");
    assert.equal(await client.getJobResult("j1"), '{"id":1}\n{"id":2}\n');
  });

  test("a response is the parsed JSON", async () => {
    const results = [{ doc_id: "d1", _id: 1, distance: 0 }];
    const { client } = mockClient(() => ({ body: { status: "searched", id: "ds", results } }));
    const response = await client.search("ds", [1]);
    assert.deepEqual(response.results, results);
  });

  test("followerSearch takes a vector or keys, and migratePartitions needs disjoint", async () => {
    const { client, calls } = mockClient();
    assert.throws(() => client.followerSearch("ds", "kw", {}), TypeError);
    assert.throws(() => client.followerSearch("ds", "kw", { vector: [1], keys: ["a"] }), TypeError);
    assert.throws(() => client.migratePartitions("ds", { name: "m" } as never), TypeError);
    assert.equal(calls.length, 0);
  });
});

describe("errors", () => {
  const answerWith = (status: number, body: unknown) => mockClient(() => ({ status, body })).client;

  test("the kind of the body selects the class of the engine", async () => {
    const client = answerWith(400, { kind: "WrongDimension", error: "expected 4, got 3" });
    const error = await client.search("ds", [1, 2, 3]).catch((e: unknown) => e);
    assert.ok(error instanceof WrongDimension);
    assert.ok(error instanceof InvalidRequest);
    assert.ok(error instanceof EngineError);
    assert.ok(error instanceof RemoteError);
    assert.ok(error instanceof Error);
    assert.equal(error.name, "WrongDimension");
    assert.equal(error.message, "expected 4, got 3");
    assert.equal(error.status, 400);
    assert.equal(error.kind, "WrongDimension");
  });

  test("each category", async () => {
    const cases: [number, string, Function, Function][] = [
      [404, "UnknownPartition", UnknownPartition, NotFound],
      [409, "PendingBacklog", PendingBacklog, ConflictingState],
      [500, "CorruptedStorage", CorruptedStorage, StorageFailure],
    ];
    for (const [status, kind, cls, category] of cases) {
      const error = await answerWith(status, { kind, error: "m" }).getDataset("ds").catch((e: unknown) => e);
      assert.ok(error instanceof cls, kind);
      assert.ok(error instanceof category, kind);
    }
  });

  test("a response without a known kind gets the generic class of its status", async () => {
    const cases: [number, unknown, Function][] = [
      [400, { error: "invalid dataset id" }, RequestRejected],
      [400, { kind: "InvalidRequest", error: "bad permission" }, RequestRejected],
      [401, { error: "missing token" }, Unauthenticated],
      [403, { error: "needs admin:*" }, Forbidden],
      [404, { error: "dataset_not_found" }, ResourceNotFound],
      [409, { error: "not completed" }, StateConflict],
      [500, { error: "boom" }, ServerFailure],
      [502, "<html>Bad Gateway</html>", UnexpectedResponse],
    ];
    for (const [status, body, cls] of cases) {
      const error = await answerWith(status, body).getDataset("ds").catch((e: unknown) => e);
      assert.ok(error instanceof cls, `${status} ${JSON.stringify(body)}`);
      assert.equal((error as RemoteError).status, status);
    }
    const proxy = await answerWith(502, "<html>Bad Gateway</html>").getDataset("ds").catch((e: unknown) => e);
    assert.equal((proxy as Error).message, "<html>Bad Gateway</html>");
    const notFound = await answerWith(404, { error: "dataset_not_found" }).getDataset("ds").catch((e: unknown) => e);
    assert.ok(notFound instanceof NotFound);
    assert.ok(!(notFound instanceof InvalidRequest));
  });

  test("ERROR_KINDS has the errors of the engine and the two of access", () => {
    assert.equal(Object.keys(ERROR_KINDS).length, 20);
    for (const [name, cls] of Object.entries(ERROR_KINDS)) {
      assert.equal(cls.name, name);
      const error = new cls(400, "m", name);
      assert.ok(error instanceof EngineError || error instanceof Unauthenticated || error instanceof Forbidden, name);
    }
  });
});

describe("jobs", () => {
  /** A server whose job `j1` answers the states of `states`, one per request, and then the last one. */
  function jobServer(states: { status: string; job?: Record<string, unknown> }[]) {
    let n = 0;
    return mockClient((call) => {
      if (call.method === "POST") return { status: 202, body: { status: "accepted", job_id: "j1" } };
      const state = states[Math.min(n++, states.length - 1)]!;
      return { body: { status: state.status, job: { id: "j1", kind: "allknn", command: [], ...state.job } } };
    });
  }

  test("waitJob returns the record when the job completes", async () => {
    const { client, calls } = jobServer([{ status: "queued" }, { status: "running" }, { status: "completed", job: { finished_at: "t" } }]);
    const record = await client.waitJob("j1", { pollInterval: 1 });
    assert.equal(record.finished_at, "t");
    assert.equal(calls.length, 3);
  });

  test("a failed job throws JobFailed with the category of its exit code", async () => {
    const { client } = jobServer([{ status: "failed", job: { error: "exit 4", exit_code: 4 } }]);
    const error = await client.runJob("allknn", { dataset: "nope" }, { pollInterval: 1 }).catch((e: unknown) => e);
    assert.ok(error instanceof JobFailed);
    assert.equal(error.jobId, "j1");
    assert.equal(error.jobKind, "allknn");
    assert.equal(error.exitCode, 4);
    assert.equal(error.category, NotFound);
    assert.equal(error.status, null);
    assert.match(error.message, /exit 4/);
    const other = await jobServer([{ status: "failed", job: { error: "cancelled by operator" } }])
      .client.waitJob("j1", { pollInterval: 1 })
      .catch((e: unknown) => e);
    assert.ok(other instanceof JobFailed);
    assert.equal(other.exitCode, null);
    assert.equal(other.category, null);
  });

  test("waitJob throws JobTimeout after its timeout", async () => {
    const { client } = jobServer([{ status: "running" }]);
    const error = await client.waitJob("j1", { pollInterval: 5, timeout: 30 }).catch((e: unknown) => e);
    assert.ok(error instanceof JobTimeout);
    assert.equal(error.jobId, "j1");
  });
});

describe("several text fields", () => {
  test("ftsearchFields sends one search per field and returns the results by field name", async () => {
    const { client, calls } = mockClient((call) => ({
      body: { status: "searched", id: call.url.pathname.split("/")[4], results: [{ doc_id: "r1", _id: 1, distance: -1 }] },
    }));
    const hits = await client.ftsearchFields({ title: "p_title", abstract: "p_abstract" }, "redes", { k: 3 });
    assert.deepEqual(Object.keys(hits), ["title", "abstract"]);
    assert.equal(hits.title![0]!.doc_id, "r1");
    assert.deepEqual(calls.map((c) => c.url.pathname).sort(), ["/api/v1/datasets/p_abstract/ftsearch", "/api/v1/datasets/p_title/ftsearch"]);
    assert.deepEqual(calls[0]!.body, { k: 3, text: "redes" });
  });
});
