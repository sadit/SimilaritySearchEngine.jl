/**
 * The shapes of the requests and the responses of the HTTP API (PLAN.md §5.17).
 *
 * The fields have the names of the wire, in snake_case, as the server manual writes them. A
 * response can carry more fields than an interface declares: the server adds a field without a
 * change of version, and an interface lists the fields that the manual documents.
 *
 * @module
 */

/** A vector: a plain array, a `Float32Array`, or any other array-like of numbers. */
export type Vector = ArrayLike<number>;

/** An item identifier: the `doc_id` that the caller gave, or the `_id` that the engine assigned. */
export type ItemId = string | number;

// --- Items ------------------------------------------------------------------------------------

/**
 * The fields that every item can carry. `doc_id`, `keywords`, `refs` and any other field become
 * the metadata of the item.
 */
export interface ItemFields {
  doc_id?: string;
  keywords?: string[];
  refs?: string[];
  [field: string]: unknown;
}

/** An item of a dense dataset. */
export interface DenseItem extends ItemFields {
  vector: Vector;
}

/** An item of a text dataset. */
export interface TextItem extends ItemFields {
  text: string;
}

/** An item of a sparse dataset: the non-zero coordinates and their values. */
export interface SparseItem extends ItemFields {
  indices: ArrayLike<number>;
  values: ArrayLike<number>;
}

export type Item = DenseItem | TextItem | SparseItem;

// --- Filters ----------------------------------------------------------------------------------

/** A comparison of one field. A field that an item does not have fails the filter. */
export interface FilterSpec {
  eq?: unknown;
  neq?: unknown;
  gt?: unknown;
  gte?: unknown;
  lt?: unknown;
  lte?: unknown;
  in?: unknown[];
  nin?: unknown[];
}

/** A filter: each field name maps to a value, compared for equality, or to a {@link FilterSpec}. */
export type Filter = Record<string, string | number | boolean | null | FilterSpec>;

// --- Datasets ---------------------------------------------------------------------------------

/** The kinds of index that the server creates. */
export type IndexKind =
  | "searchgraph"
  | "exhaustive_search"
  | "parallel_exhaustive_search"
  | "bm25_invfile"
  | "invfile"
  | "sparse_invfile";

/** The types of a declared metadata field. */
export type MetaType = "string" | "int64" | "float64" | "bool" | "timestamp";

export interface MetaField {
  name: string;
  type: MetaType;
}

/** The storage of a dense dataset. The server manual lists the fields of each kind. */
export type Storage =
  | { kind: "float32" }
  | {
      kind: "quantized";
      bits?: 8 | 4 | 2;
      range?: "global" | "per_vector";
      minmax?: [number, number] | null;
      samplesize?: number;
      rotation?: "none" | "qr" | "hadamard";
      keep_vectors?: boolean;
    }
  | {
      kind: "sketched";
      hyperplanes?: number;
      bits?: 1 | 2 | 4 | 8;
      rerank?: number;
      keep_vectors?: boolean;
      center?: boolean;
      samplesize?: number;
      hsel?: number;
      henc?: number;
      entquantile?: number;
      anchorpolicy?: string;
    };

/** The body of `POST /api/v1/datasets`. A field that is absent takes the default of the server. */
export interface CreateDatasetOptions {
  id?: string;
  index_kind?: IndexKind;
  distance?: string;
  /** Required for `sparse_invfile`, and refused for the other kinds. */
  dimension?: number;
  meta_schema?: MetaField[];
  edit_correction?: boolean;
  partition_by?: string;
  partition_graph_min?: number;
  storage?: Storage;
  /** The largest `k` of a search. 100 when it is absent. */
  max_k?: number;
}

/** The parameters of a beam search. */
export interface BeamSearchParams {
  bsize: number;
  delta: number;
  maxvisits: number;
}

/** What `GET /api/v1/datasets/{id}` reports. The live fields are present for a loaded dataset. */
export interface DatasetInfo {
  id: string;
  index_kind: string;
  distance: string;
  created_at: string | null;
  loaded?: boolean;
  edit_correction?: boolean;
  max_k?: number;
  doc_count?: number;
  tombstone_count?: number;
  tombstone_ratio?: number;
  meta_schema?: MetaField[] | null;
  partition_by?: string | null;
  partition_graph_min?: number | null;
  beamsearch_baseline?: BeamSearchParams | null;
  storage?: Record<string, unknown> | null;
  encoder?: Record<string, unknown> | null;
  followers?: FollowerStatus[];
  [field: string]: unknown;
}

export interface DatasetCreated {
  status: "created";
  id: string;
}

export interface DatasetList {
  status: "ok";
  datasets: DatasetInfo[];
  total: number;
  offset: number;
  limit: number;
}

/** The answer of a route that changes the state of a dataset: `deleted`, `unloaded`, `reloaded`. */
export interface DatasetStatus {
  status: string;
  id: string;
}

export interface LogEntry {
  operation: string;
  timestamp: string;
  details: Record<string, unknown>;
}

export interface LogPage {
  status: "ok";
  entries: LogEntry[];
  total: number;
  offset: number;
  limit: number;
}

export interface VocabOptions {
  /** Reads the documents of the dataset to count the tokens outside the vocabulary. */
  scan?: boolean;
  /** The number of documents that `scan` reads. 0 reads every document. */
  sample?: number;
}

export interface VocabReport {
  id: string;
  vocsize: number;
  trainsize: number;
  numtokens: number;
  avgdoclen: number;
  top_tokens: { token: string; ndocs: number; occs: number }[];
  queries: number;
  query_tokens: number;
  query_oov_tokens: number;
  query_oov_rate: number;
  live_tokens?: number;
  live_oov_tokens?: number;
  live_oov_rate?: number;
  live_documents_scanned?: number;
  live_documents_total?: number;
  live_sampled?: boolean;
  [field: string]: unknown;
}

/** One partition: its name, the number of items appended to it, and whether it has a graph. */
export interface PartitionInfo {
  name: string;
  items: number;
  graph: boolean;
}

export interface PartitionList {
  status: "ok";
  id: string;
  partition_by: string;
  partition_graph_min: number | null;
  partitions: PartitionInfo[];
}

// --- Items and searches -----------------------------------------------------------------------

export interface AppendResponse {
  status: "appended";
  id: string;
  inserted: number;
}

export interface SearchOptions {
  /** 10 when it is absent, and at most `max_k` of the dataset. */
  k?: number;
  filter?: Filter;
  /** The parameters of the beam search of a `searchgraph` dataset for this request. */
  beamsearch_overrides?: Partial<BeamSearchParams>;
  /** One partition of a dense dataset with `partition_by`. */
  partition?: string;
}

export interface FtsearchOptions {
  k?: number;
  filter?: Filter;
  partition?: string;
}

/** One result. `distance` of a text search is the BM25 score with its sign inverted. */
export interface SearchHit {
  doc_id: string | null;
  _id: number;
  distance: number;
}

export interface SearchResponse {
  status: "searched";
  id: string;
  results: SearchHit[];
  /** Present when a filtered search could not fill `k`. */
  insufficient_results?: true;
}

export interface HybridSearchOptions {
  vector?: Vector;
  text?: string;
  k?: number;
}

export interface HybridHit {
  doc_id: string | null;
  _id: number;
  score: number;
}

export interface HybridSearchResponse {
  status: "searched";
  results: HybridHit[];
}

/** A stored record: its fixed fields and, merged into it, the metadata of the item. */
export interface StoredRecord {
  doc_id: string | null;
  _id: number;
  keywords: string[];
  refs: string[];
  schema_version: number;
  [field: string]: unknown;
}

export interface FetchResponse {
  status: "fetched";
  id: string;
  /** The records that exist. An identifier that does not exist has no entry. */
  results: StoredRecord[];
}

export interface ExistsEntry {
  /** The identifier that was asked about. */
  id: string;
  doc_id: string | null;
  _id: number | null;
  exists: boolean;
  deleted: boolean;
}

export interface ExistsResponse {
  status: "ok";
  id: string;
  results: ExistsEntry[];
}

export interface DeleteItemResponse {
  status: "soft_deleted";
  id: string;
  _id: number;
}

export interface CalibrateOptions {
  minrecall?: number;
  numqueries?: number;
  ksearch?: number;
  queries?: Vector[];
  tradeoff?: unknown;
  transition?: unknown;
  /** Calibrates the graph of one partition of a dense dataset. */
  partition?: string;
}

export interface Calibration {
  minrecall: number;
  recall: number;
  recall_lo: number;
  recall_hi: number;
  confidence: number;
  numqueries: number;
  distance_evaluations: number;
  heldout: boolean;
  [field: string]: unknown;
}

export interface CalibrateResponse {
  status: "calibrated";
  id: string;
  partition?: string;
  follower?: string;
  baseline: BeamSearchParams;
  calibration: Calibration;
}

export interface PostingListOptions {
  /** A keyword follower of the dataset. Without it, the posting list is the dataset's own. */
  follower?: string;
  /** Returns the first `limit` ids. */
  limit?: number;
}

export interface PostingList {
  id: string;
  follower?: string;
  key: string;
  /** The `_id`s of the items that hold the key, in ascending order. */
  ids: number[];
}

// --- Followers --------------------------------------------------------------------------------

/** The policy of a follower: when it processes the new items of its leader. */
export type FollowerPolicy =
  | "on_index"
  | "on_write"
  | "manual"
  | { periodic: { seconds?: number; items?: number } };

/** A kind or a key source of a plugin, with its options: `{ name: "Decades.decade", field: "year" }`. */
export interface PluginName {
  name: string;
  [option: string]: unknown;
}

/** The body of `POST /api/v1/datasets/{id}/followers`. */
export interface CreateFollowerOptions {
  name: string;
  /** `partition`, `keyword`, or a kind of a plugin. */
  kind: string | PluginName;
  /** `keywords`, `refs`, `doc_id`, `meta:<field>`, or a key source of a plugin. */
  source: string | PluginName;
  disjoint?: boolean;
  graph_min?: number;
  distance?: "jaccard" | "dice" | "intersection" | "cosine_set";
  policy?: FollowerPolicy;
}

/** The status of a follower. The fields are those of the engine's `FollowerStatus`. */
export interface FollowerStatus {
  name: string;
  leader: string;
  kind: string;
  source: string;
  policy: string;
  plugin: string | null;
  position: number;
  leader_indexed: number;
  lag: number;
  synchronized: boolean;
  stale: boolean;
  missing: boolean;
  open: boolean;
  stats: Record<string, number> | null;
  /** The message of the last failure of the follower. */
  error?: string;
  [field: string]: unknown;
}

export interface FollowerList {
  id: string;
  followers: FollowerStatus[];
}

export interface FollowerDeleted {
  status: "deleted";
  id: string;
  follower: string;
}

export interface FollowerSynced extends FollowerStatus {
  status: "synchronized";
  /** The number of items that this call processed. */
  processed: number;
}

export interface FollowerPartitions {
  id: string;
  follower: string;
  partitions: PartitionInfo[];
}

/** The search of a follower: a vector in one partition, or keys in a keyword follower. */
export interface FollowerSearchOptions {
  vector?: Vector;
  keys?: string[];
  k?: number;
  /** The number of keys of the query that an item must hold to be a candidate. 1 by default. */
  t?: number;
  filter?: Filter;
  minrecall?: number;
}

export interface FollowerCalibrateOptions {
  minrecall?: number;
  numqueries?: number;
  ksearch?: number;
}

export interface MigratePartitionsOptions {
  name: string;
  disjoint: boolean;
}

export interface Group {
  leader: DatasetInfo;
  followers: FollowerStatus[];
}

export interface PluginInfo {
  name: string;
  path?: string;
  package?: string;
  key_sources: string[];
  follower_kinds: string[];
}

export interface PluginList {
  status: "ok";
  plugins: PluginInfo[];
}

// --- Jobs -------------------------------------------------------------------------------------

/** The kinds of job that the server builds from parameters. */
export type JobKind = "allknn" | "fft" | "neardup" | "hsp" | "dump" | "load";

/**
 * The body of `POST /api/v1/jobs/{kind}`: `dataset` and `k` for `allknn`, `dataset` and `k` for
 * `fft`, `dataset` and `epsilon` for `neardup`, `dataset`, `queries` and `k` for `hsp`, `dataset`
 * for `dump`, and `dataset` and `bundle` for `load`. `command`, the arguments of the server's
 * command line, submits any other kind.
 */
export interface JobParams {
  dataset?: string;
  k?: number;
  epsilon?: number;
  queries?: string;
  bundle?: string;
  command?: string[];
  [field: string]: unknown;
}

export interface JobAccepted {
  status: "accepted";
  job_id: string;
}

export type JobState = "queued" | "running" | "blocked" | "completed" | "failed";

/** The record of a job. A failed job has `error`, and `exit_code` when its process ended. */
export interface JobRecord {
  id: string;
  kind: string;
  command: string[];
  created_at: string;
  started_at?: string;
  finished_at?: string;
  error?: string;
  exit_code?: number;
  [field: string]: unknown;
}

export interface JobInfo {
  status: JobState;
  job: JobRecord;
}

export interface ListJobsOptions {
  status?: JobState;
  kind?: string;
  offset?: number;
  limit?: number;
}

export interface JobList {
  status: "ok";
  jobs: { id: string; kind: string; status: JobState; [field: string]: unknown }[];
  total: number;
  offset: number;
  limit: number;
}

/** The answer of `block`, `resume`, `kill` and `cancel`. */
export interface JobControl {
  status: string;
  job_id: string;
}

/** The options of `waitJob` and `runJob`. They are options of the client, not of the wire. */
export interface WaitOptions {
  /** Milliseconds between two requests for the state of the job. 1000 by default. */
  pollInterval?: number;
  /** Milliseconds to wait before throwing `JobTimeout`. No limit by default. */
  timeout?: number;
}

// --- Administration ---------------------------------------------------------------------------

export interface CreateTokenOptions {
  user?: string;
  /** Permissions of the form `operation:dataset`: `read:*`, `write:papers`, `admin:*`. */
  permissions?: string[];
  /** An ISO 8601 timestamp. */
  expires_at?: string;
}

export interface TokenCreated {
  status: "created";
  token: string;
  permissions: string[];
}

export interface TokenInfo {
  token: string;
  user: string;
  permissions: string[];
  created_at: string;
  expires_at: string | null;
}

export interface TokenList {
  status: "ok";
  tokens: TokenInfo[];
}

export interface ReloadOptions {
  max_k?: number;
  edit_correction?: boolean;
}

export interface JobsGcResponse {
  status: "ok";
  jobs_purged: string[];
}
