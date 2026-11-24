/**
 * The client of the HTTP API of `similarity-search-serve` (PLAN.md §5.17).
 *
 * @module
 */

import { JobFailed, JobTimeout, responseError } from "./errors.js";
import type {
  AppendResponse,
  CalibrateOptions,
  CalibrateResponse,
  CreateDatasetOptions,
  CreateFollowerOptions,
  CreateTokenOptions,
  DatasetCreated,
  DatasetInfo,
  DatasetList,
  DatasetStatus,
  DeleteItemResponse,
  ExistsResponse,
  FetchResponse,
  FollowerCalibrateOptions,
  FollowerDeleted,
  FollowerList,
  FollowerPartitions,
  FollowerSearchOptions,
  FollowerStatus,
  FollowerSynced,
  FtsearchOptions,
  Group,
  HybridSearchOptions,
  HybridSearchResponse,
  Item,
  ItemId,
  JobAccepted,
  JobControl,
  JobInfo,
  JobKind,
  JobList,
  JobParams,
  JobRecord,
  JobsGcResponse,
  ListJobsOptions,
  LogPage,
  MigratePartitionsOptions,
  PartitionList,
  PluginList,
  PostingList,
  PostingListOptions,
  ReloadOptions,
  SearchHit,
  SearchOptions,
  SearchResponse,
  TokenCreated,
  TokenList,
  Vector,
  VocabOptions,
  VocabReport,
  WaitOptions,
} from "./types.js";

/** The options of {@link SimilaritySearchClient}. */
export interface ClientOptions {
  /** Sent as `Authorization: Bearer <token>` with every request. */
  token?: string;
  /** Milliseconds to wait for a response. 60,000 by default, and 0 waits without a limit. */
  timeout?: number;
  /** A function with the signature of `fetch`, used in place of the global one. */
  fetch?: typeof globalThis.fetch;
}

type Query = Record<string, string | number | boolean | undefined>;

interface RequestOptions {
  body?: unknown;
  query?: Query;
  /** Returns the text of the response instead of its parsed JSON. */
  text?: boolean;
}

const API = "/api/v1";

/** One path segment: a dataset id, a follower name (`name=value`), a job id or a token. */
const seg = (s: string): string => encodeURIComponent(s);

/** A vector as a JSON array. `JSON.stringify` writes a `Float32Array` as an object, not an array. */
const asArray = (v: ArrayLike<number>): number[] => (Array.isArray(v) ? v : Array.from(v));

/** An item with its vector, its indices and its values as JSON arrays. */
function wireItem(item: Item): Record<string, unknown> {
  const out: Record<string, unknown> = { ...item };
  for (const field of ["vector", "indices", "values"]) {
    const v = out[field];
    if (v !== undefined && v !== null && typeof v === "object" && !Array.isArray(v)) {
      out[field] = Array.from(v as ArrayLike<number>);
    }
  }
  return out;
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * A client of one `similarity-search-serve` process, with one `async` method per route.
 *
 * A method returns the parsed JSON of the response, with the field names of the wire. A response
 * with an error status throws the {@link RemoteError} that its `kind` names: `NotFound`,
 * `WrongDimension`, `PendingBacklog`, and so on. An error of the connection is the error of
 * `fetch`.
 *
 * ```ts
 * const client = new SimilaritySearchClient("http://127.0.0.1:8080", { token });
 * await client.createDataset({ id: "docs", index_kind: "searchgraph", distance: "NormCosine" });
 * await client.append("docs", [{ doc_id: "a", vector: new Float32Array([0.1, 0.2]) }]);
 * const { results } = await client.search("docs", [0.1, 0.2], { k: 5 });
 * ```
 */
export class SimilaritySearchClient {
  /** The root of the server, without `/api/v1` and without a final `/`. */
  readonly url: string;
  readonly timeout: number;
  readonly #token: string | undefined;
  readonly #fetch: typeof globalThis.fetch;

  constructor(url: string = "http://127.0.0.1:8080", options: ClientOptions = {}) {
    this.url = url.replace(/\/+$/, "");
    this.timeout = options.timeout ?? 60_000;
    this.#token = options.token;
    this.#fetch = options.fetch ?? globalThis.fetch;
  }

  /**
   * Sends one request and returns the parsed JSON of a 2xx response, or its text with `text`.
   * Any other status throws the error that `responseError` selects.
   */
  async request<T>(method: string, path: string, options: RequestOptions = {}): Promise<T> {
    const url = new URL(this.url + path);
    for (const [name, value] of Object.entries(options.query ?? {})) {
      if (value !== undefined) url.searchParams.set(name, String(value));
    }
    const headers: Record<string, string> = { accept: "application/json" };
    if (this.#token !== undefined) headers.authorization = `Bearer ${this.#token}`;
    const init: RequestInit = { method, headers };
    if (options.body !== undefined) {
      headers["content-type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }
    if (this.timeout > 0) init.signal = AbortSignal.timeout(this.timeout);
    const response = await this.#fetch(url, init);
    const text = await response.text();
    if (!response.ok) throw responseError(response.status, text);
    if (options.text) return text as T;
    return (text === "" ? undefined : JSON.parse(text)) as T;
  }

  // --- Control --------------------------------------------------------------------------------

  /** `GET /healthz`: the process answers HTTP. */
  healthz(): Promise<{ status: string }> {
    return this.request("GET", "/healthz");
  }

  /** `GET /readyz`: the process is ready to serve. */
  readyz(): Promise<{ status: string }> {
    return this.request("GET", "/readyz");
  }

  /** `GET /metrics`: the metrics in the text format of Prometheus. */
  metrics(): Promise<string> {
    return this.request("GET", "/metrics", { text: true });
  }

  // --- Datasets -------------------------------------------------------------------------------

  /** `POST /api/v1/datasets`. Without `id`, the server generates one and returns it. */
  createDataset(options: CreateDatasetOptions = {}): Promise<DatasetCreated> {
    return this.request("POST", `${API}/datasets`, { body: options });
  }

  /** `GET /api/v1/datasets`: every dataset on disk, loaded or not. */
  listDatasets(options: { offset?: number; limit?: number } = {}): Promise<DatasetList> {
    return this.request("GET", `${API}/datasets`, { query: options });
  }

  /** `GET /api/v1/datasets/{id}`. */
  getDataset(id: string): Promise<DatasetInfo> {
    return this.request("GET", `${API}/datasets/${seg(id)}`);
  }

  /** `DELETE /api/v1/datasets/{id}`. A dataset with followers throws `HasFollowers`. */
  deleteDataset(id: string): Promise<DatasetStatus> {
    return this.request("DELETE", `${API}/datasets/${seg(id)}`);
  }

  /** `GET /api/v1/datasets/{id}/log`: the operation log. It requires the permission `admin`. */
  getLog(id: string, options: { offset?: number; limit?: number } = {}): Promise<LogPage> {
    return this.request("GET", `${API}/datasets/${seg(id)}/log`, { query: options });
  }

  /** `GET /api/v1/datasets/{id}/vocab`: what the vocabulary of a text dataset covers. */
  getVocab(id: string, options: VocabOptions = {}): Promise<VocabReport> {
    return this.request("GET", `${API}/datasets/${seg(id)}/vocab`, { query: { ...options } });
  }

  /** `GET /api/v1/datasets/{id}/partitions`: the partitions of a dataset with `partition_by`. */
  getPartitions(id: string): Promise<PartitionList> {
    return this.request("GET", `${API}/datasets/${seg(id)}/partitions`);
  }

  // --- Items and searches ---------------------------------------------------------------------

  /** `POST /api/v1/datasets/{id}/append`: stores and indexes the items. */
  append(id: string, items: Item[]): Promise<AppendResponse> {
    return this.request("POST", `${API}/datasets/${seg(id)}/append`, { body: { items: items.map(wireItem) } });
  }

  /** `POST /api/v1/datasets/{id}/search`: a search of a dense or a sparse dataset. */
  search(id: string, vector: Vector, options: SearchOptions = {}): Promise<SearchResponse> {
    return this.request("POST", `${API}/datasets/${seg(id)}/search`, {
      body: { ...options, vector: asArray(vector) },
    });
  }

  /** `POST /api/v1/datasets/{id}/ftsearch`: a search of a text dataset. */
  ftsearch(id: string, text: string, options: FtsearchOptions = {}): Promise<SearchResponse> {
    return this.request("POST", `${API}/datasets/${seg(id)}/ftsearch`, { body: { ...options, text } });
  }

  /**
   * `POST /api/v1/search/hybrid`: the reciprocal rank fusion of a search of `denseIndex` with
   * `vector` and a text search of `lexicalIndex` with `text`.
   */
  hybridSearch(denseIndex: string, lexicalIndex: string, options: HybridSearchOptions = {}): Promise<HybridSearchResponse> {
    const { vector, ...rest } = options;
    return this.request("POST", `${API}/search/hybrid`, {
      body: {
        dense_index: denseIndex,
        lexical_index: lexicalIndex,
        ...rest,
        vector: vector === undefined ? undefined : asArray(vector),
      },
    });
  }

  /** `POST /api/v1/datasets/{id}/fetch`: the records of the items, by `doc_id` or `_id`. */
  fetch(id: string, ids: ItemId[]): Promise<FetchResponse> {
    return this.request("POST", `${API}/datasets/${seg(id)}/fetch`, { body: { ids } });
  }

  /**
   * `GET /api/v1/datasets/{id}/exists`: whether each identifier exists. The identifiers travel
   * in the query, separated by commas, so a `doc_id` with a comma cannot be asked about.
   */
  exists(id: string, ids: ItemId[]): Promise<ExistsResponse> {
    return this.request("GET", `${API}/datasets/${seg(id)}/exists`, { query: { ids: ids.join(",") } });
  }

  /** `POST /api/v1/datasets/{id}/delete`: deletes the item `_id`, the integer of a search or a fetch. */
  deleteItem(id: string, _id: number): Promise<DeleteItemResponse> {
    return this.request("POST", `${API}/datasets/${seg(id)}/delete`, { body: { _id } });
  }

  /** `POST /api/v1/datasets/{id}/calibrate`: tunes the beam search of a graph for a recall. */
  calibrate(id: string, options: CalibrateOptions = {}): Promise<CalibrateResponse> {
    const { queries, ...rest } = options;
    return this.request("POST", `${API}/datasets/${seg(id)}/calibrate`, {
      body: { ...rest, queries: queries === undefined ? undefined : queries.map(asArray) },
    });
  }

  /**
   * `GET /api/v1/datasets/{id}/postings`: the `_id`s of the items that hold `key`, a token of a
   * text dataset or a dimension of a sparse one. With `follower`, a key of a keyword follower.
   */
  postingList(id: string, key: string | number, options: PostingListOptions = {}): Promise<PostingList> {
    const path =
      options.follower === undefined
        ? `${API}/datasets/${seg(id)}/postings`
        : `${API}/datasets/${seg(id)}/followers/${seg(options.follower)}/postings`;
    return this.request("GET", path, { query: { key: String(key), limit: options.limit } });
  }

  // --- Followers ------------------------------------------------------------------------------

  /** `POST /api/v1/datasets/{id}/followers`: creates a follower and processes the leader. */
  createFollower(id: string, options: CreateFollowerOptions): Promise<FollowerStatus & { status: "created" }> {
    return this.request("POST", `${API}/datasets/${seg(id)}/followers`, { body: options });
  }

  /** `GET /api/v1/datasets/{id}/followers`: the status of every follower. */
  listFollowers(id: string): Promise<FollowerList> {
    return this.request("GET", `${API}/datasets/${seg(id)}/followers`);
  }

  /** `GET /api/v1/datasets/{id}/followers/{name}`. */
  getFollower(id: string, name: string): Promise<FollowerStatus> {
    return this.request("GET", this.#follower(id, name));
  }

  /** `DELETE /api/v1/datasets/{id}/followers/{name}`. */
  deleteFollower(id: string, name: string): Promise<FollowerDeleted> {
    return this.request("DELETE", this.#follower(id, name));
  }

  /** `POST .../followers/{name}/sync`: processes the new items of the leader now. */
  syncFollower(id: string, name: string): Promise<FollowerSynced> {
    return this.request("POST", this.#follower(id, name, "/sync"));
  }

  /** `POST .../followers/{name}/rebuild`: builds the follower again from every indexed item. */
  rebuildFollower(id: string, name: string): Promise<FollowerStatus & { status: "rebuilt" }> {
    return this.request("POST", this.#follower(id, name, "/rebuild"));
  }

  /** `GET .../followers/{name}/partitions`: the partitions of a partition follower. */
  followerPartitions(id: string, name: string): Promise<FollowerPartitions> {
    return this.request("GET", this.#follower(id, name, "/partitions"));
  }

  /**
   * `POST .../followers/{follower}/search`. `follower` is `name=value` with a `vector`, to search
   * one partition of a dense dataset, or the name of a keyword follower with `keys`.
   */
  followerSearch(id: string, follower: string, options: FollowerSearchOptions): Promise<SearchResponse> {
    const { vector, ...rest } = options;
    if ((vector === undefined) === (rest.keys === undefined)) {
      throw new TypeError(
        "followerSearch takes a vector (a partition, name=value) or keys (a keyword follower), and not both",
      );
    }
    return this.request("POST", this.#follower(id, follower, "/search"), {
      body: { ...rest, vector: vector === undefined ? undefined : asArray(vector) },
    });
  }

  /** `POST .../followers/{name=value}/ftsearch`: a text search in one partition. */
  followerFtsearch(id: string, follower: string, text: string, options: { k?: number; filter?: FtsearchOptions["filter"] } = {}): Promise<SearchResponse> {
    return this.request("POST", this.#follower(id, follower, "/ftsearch"), { body: { ...options, text } });
  }

  /** `POST .../followers/{name=value}/calibrate`: calibrates the graph of one partition. */
  followerCalibrate(id: string, follower: string, options: FollowerCalibrateOptions = {}): Promise<CalibrateResponse> {
    return this.request("POST", this.#follower(id, follower, "/calibrate"), { body: options });
  }

  /**
   * `POST /api/v1/datasets/{id}/migrate-partitions`: moves the partitions of `partition_by` to a
   * partition follower. `disjoint` has no default.
   */
  migratePartitions(id: string, options: MigratePartitionsOptions): Promise<FollowerStatus & { status: "migrated" }> {
    if (typeof options.disjoint !== "boolean") {
      throw new TypeError("migratePartitions needs disjoint: true or disjoint: false");
    }
    return this.request("POST", `${API}/datasets/${seg(id)}/migrate-partitions`, { body: options });
  }

  /** `GET /api/v1/datasets/{id}/group`: the leader and the status of every follower. */
  getGroup(id: string): Promise<Group> {
    return this.request("GET", `${API}/datasets/${seg(id)}/group`);
  }

  /** `GET /api/v1/plugins`: the plugins that the server loaded. */
  listPlugins(): Promise<PluginList> {
    return this.request("GET", `${API}/plugins`);
  }

  #follower(id: string, name: string, rest: string = ""): string {
    return `${API}/datasets/${seg(id)}/followers/${seg(name)}${rest}`;
  }

  // --- Jobs -----------------------------------------------------------------------------------

  /** `POST /api/v1/jobs/{kind}`: submits a job. The job runs in a separate process on the server. */
  submitJob(kind: JobKind | string, params: JobParams): Promise<JobAccepted> {
    return this.request("POST", `${API}/jobs/${seg(kind)}`, { body: params });
  }

  /** `GET /api/v1/jobs/{jobId}`: the state of a job and its record. */
  getJob(jobId: string): Promise<JobInfo> {
    return this.request("GET", `${API}/jobs/${seg(jobId)}`);
  }

  /**
   * `GET /api/v1/jobs/{jobId}/result`: the text of the file that the job wrote. It is JSON lines
   * for `allknn` and `hsp`, one JSON object for `fft` and `neardup`, and for `dump` an object
   * with `bundle_dir` and `manifest`.
   */
  getJobResult(jobId: string): Promise<string> {
    return this.request("GET", `${API}/jobs/${seg(jobId)}/result`, { text: true });
  }

  /** `GET /api/v1/jobs`. */
  listJobs(options: ListJobsOptions = {}): Promise<JobList> {
    return this.request("GET", `${API}/jobs`, { query: { ...options } });
  }

  /** `POST /api/v1/jobs/{jobId}/block`: keeps a queued job from starting. */
  blockJob(jobId: string): Promise<JobControl> {
    return this.request("POST", `${API}/jobs/${seg(jobId)}/block`);
  }

  /** `POST /api/v1/jobs/{jobId}/resume`: queues a blocked job again. */
  resumeJob(jobId: string): Promise<JobControl> {
    return this.request("POST", `${API}/jobs/${seg(jobId)}/resume`);
  }

  /** `POST /api/v1/jobs/{jobId}/kill`: stops the process of a running job. */
  killJob(jobId: string): Promise<JobControl> {
    return this.request("POST", `${API}/jobs/${seg(jobId)}/kill`);
  }

  /** `DELETE /api/v1/jobs/{jobId}`: cancels a job. */
  cancelJob(jobId: string): Promise<JobControl> {
    return this.request("DELETE", `${API}/jobs/${seg(jobId)}`);
  }

  /**
   * Asks for the state of a job every `pollInterval` milliseconds until it ends, and returns its
   * record. A failed job throws {@link JobFailed}. After `timeout` milliseconds it throws
   * {@link JobTimeout}, and the job keeps running on the server.
   */
  async waitJob(jobId: string, options: WaitOptions = {}): Promise<JobRecord> {
    const pollInterval = options.pollInterval ?? 1000;
    const timeout = options.timeout ?? Infinity;
    const start = Date.now();
    for (;;) {
      const info = await this.getJob(jobId);
      if (info.status === "completed") return info.job;
      if (info.status === "failed") {
        const code = typeof info.job.exit_code === "number" ? info.job.exit_code : null;
        throw new JobFailed(jobId, String(info.job.kind ?? ""), code, String(info.job.error ?? "the job failed"));
      }
      const elapsed = Date.now() - start;
      if (elapsed >= timeout) throw new JobTimeout(jobId, timeout);
      await sleep(Math.min(pollInterval, timeout - elapsed));
    }
  }

  /** Submits a job and waits for it, as {@link waitJob}. Returns the record of the job. */
  async runJob(kind: JobKind | string, params: JobParams, options: WaitOptions = {}): Promise<JobRecord> {
    const { job_id } = await this.submitJob(kind, params);
    return this.waitJob(job_id, options);
  }

  // --- Several text fields ------------------------------------------------------------------

  /**
   * Searches `text` in every dataset of `fields`, an object from a field name to a dataset id.
   * The requests are sent at the same time. Returns the results of each dataset by field name.
   * The items of a record have the same `doc_id` in every field dataset, so `doc_id` identifies
   * the record across the fields.
   */
  async ftsearchFields(fields: Record<string, string>, text: string, options: FtsearchOptions = {}): Promise<Record<string, SearchHit[]>> {
    const names = Object.keys(fields);
    const responses = await Promise.all(names.map((name) => this.ftsearch(fields[name] as string, text, options)));
    return Object.fromEntries(names.map((name, i) => [name, (responses[i] as SearchResponse).results]));
  }

  // --- Administration -------------------------------------------------------------------------

  /** `POST /api/v1/admin/tokens`. The token is returned once, in this response. */
  createToken(options: CreateTokenOptions = {}): Promise<TokenCreated> {
    return this.request("POST", `${API}/admin/tokens`, {
      body: { user: options.user ?? "anonymous", permissions: options.permissions ?? [], expires_at: options.expires_at },
    });
  }

  /** `GET /api/v1/admin/tokens`: every stored token, expired or not. */
  listTokens(): Promise<TokenList> {
    return this.request("GET", `${API}/admin/tokens`);
  }

  /** `POST /api/v1/admin/tokens/prune`: removes the expired tokens. */
  pruneTokens(): Promise<{ status: "ok"; pruned: number }> {
    return this.request("POST", `${API}/admin/tokens/prune`);
  }

  /** `DELETE /api/v1/admin/tokens/{token}`. */
  revokeToken(token: string): Promise<{ status: "revoked" }> {
    return this.request("DELETE", `${API}/admin/tokens/${seg(token)}`);
  }

  /** `POST /api/v1/admin/jobs/gc`: removes the finished jobs older than `retention_seconds`. */
  jobsGc(options: { retention_seconds?: number } = {}): Promise<JobsGcResponse> {
    return this.request("POST", `${API}/admin/jobs/gc`, { body: options });
  }

  /** `POST /api/v1/admin/datasets/{id}/unload`: closes a dataset without stopping the server. */
  unloadDataset(id: string): Promise<DatasetStatus> {
    return this.request("POST", `${API}/admin/datasets/${seg(id)}/unload`);
  }

  /** `POST /api/v1/admin/datasets/{id}/reload`: opens a dataset again, with a new `max_k` or `edit_correction`. */
  reloadDataset(id: string, options: ReloadOptions = {}): Promise<DatasetStatus> {
    const body = Object.keys(options).length === 0 ? undefined : options;
    return this.request("POST", `${API}/admin/datasets/${seg(id)}/reload`, { body });
  }
}
