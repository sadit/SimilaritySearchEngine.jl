/**
 * The errors of the client (PLAN.md §5.17).
 *
 * Every error of a server response is a subclass of {@link RemoteError}. The classes have the
 * names and the categories of the engine's errors, as in the Julia client: the server sends the
 * name of the engine's error in the `kind` field of the body, and the client throws the class of
 * the same name. An error of the connection, such as a refused connection or a timeout, is not
 * wrapped: it is the error that `fetch` throws.
 *
 * @module
 */

/**
 * Every error that this package throws for a response of the server, and for a job that did not
 * finish.
 *
 * `status` is the HTTP status of the response, and `null` for {@link JobFailed} and
 * {@link JobTimeout}. `kind` is the `kind` field of the body, the name of the engine's error, and
 * `null` when the body has none. `message` is the `error` field of the body.
 */
export class RemoteError extends Error {
  readonly status: number | null;
  readonly kind: string | null;

  constructor(status: number | null, message: string, kind: string | null = null) {
    super(message);
    this.name = new.target.name;
    this.status = status;
    this.kind = kind;
  }
}

/**
 * An error of the engine, reported by the server. It has the four categories of the engine:
 * {@link InvalidRequest} (HTTP 400), {@link NotFound} (404), {@link ConflictingState} (409) and
 * {@link StorageFailure} (500). The category is the contract: `e instanceof NotFound` is true
 * for every error of that category.
 */
export class EngineError extends RemoteError {}

/** The request cannot be served as written. Sending it again unchanged fails again. HTTP 400. */
export class InvalidRequest extends EngineError {}
/** Something that the request names does not exist. HTTP 404. */
export class NotFound extends EngineError {}
/** The request is valid, but the dataset is not in a state that can serve it. HTTP 409. */
export class ConflictingState extends EngineError {}
/** The server read stored data that it cannot interpret. HTTP 500 with an engine `kind`. */
export class StorageFailure extends EngineError {}

/** An item or a query of the wrong kind for the dataset. */
export class PayloadMismatch extends InvalidRequest {}
/** A vector whose length is not the dimension of the dataset. */
export class WrongDimension extends InvalidRequest {}
/** An index kind that the server does not know. */
export class UnknownBackend extends InvalidRequest {}
/** An option that does not apply, is missing, or has a value that the server cannot use. */
export class InvalidOption extends InvalidRequest {}
/** An operation that this kind of dataset does not implement. */
export class UnsupportedOperation extends InvalidRequest {}
/** A 400 response without an engine `kind`: the server refused the request before the engine. */
export class RequestRejected extends InvalidRequest {}

/** A linguistic profile that the server does not have. */
export class ProfileNotInstalled extends NotFound {}
/** A partition name that the dataset does not have. The message lists the closest names. */
export class UnknownPartition extends NotFound {}
/** A follower name that the leader does not have. The message lists the closest names. */
export class UnknownFollower extends NotFound {}
/** A follower that needs a plugin that the server has not loaded. */
export class PluginNotLoaded extends NotFound {}
/** A 404 response without an engine `kind`, for example a dataset that is not loaded. */
export class ResourceNotFound extends NotFound {}

/** Items are appended but not indexed yet. */
export class PendingBacklog extends ConflictingState {}
/** Nothing was ever appended, so there is nothing to index. */
export class NothingStaged extends ConflictingState {}
/** A text dataset with a profile and no index. */
export class NoIndex extends ConflictingState {}
/** A text dataset without a vocabulary, a partition that was not calibrated, or an encoder that was not fitted. */
export class NotTrained extends ConflictingState {}
/** An operation over the whole dataset, on a dataset without items. */
export class EmptyProject extends ConflictingState {}
/** The server closed the dataset while it served the request, for example to reload it. */
export class ProjectClosed extends ConflictingState {}
/** A follower built before its leader was rebuilt. Rebuilding the follower makes it usable again. */
export class StaleFollower extends ConflictingState {}
/** A request to delete a dataset that has followers. The followers are deleted first. */
export class HasFollowers extends ConflictingState {}
/** A 409 response without an engine `kind`. */
export class StateConflict extends ConflictingState {}

/** Stored data that the server cannot interpret. */
export class CorruptedStorage extends StorageFailure {}

/** The server refused the token of the request: HTTP 401 or 403. */
export class AccessDenied extends RemoteError {}
/** The request has no token, or its token is not valid. HTTP 401. */
export class Unauthenticated extends AccessDenied {}
/** The token does not have the permission that the route requires. HTTP 403. */
export class Forbidden extends AccessDenied {}

/** A 500 response without an engine `kind`: an error of the server that is not one of the engine's. */
export class ServerFailure extends RemoteError {}

/** A response with a status that this package does not map, for example a 502 of a proxy. */
export class UnexpectedResponse extends RemoteError {}

/** One of the four categories of the engine's errors. */
export type ErrorCategory =
  | typeof InvalidRequest
  | typeof NotFound
  | typeof ConflictingState
  | typeof StorageFailure;

/**
 * A job that ended in the `failed` state.
 *
 * `exitCode` is the exit status of the process of the job, or `null` when the server did not
 * record it. The command line of the server exits with one code per category of engine error,
 * and `category` is that category: 2 is {@link InvalidRequest}, 4 is {@link NotFound}, 5 is
 * {@link ConflictingState} and 70 is {@link StorageFailure}. Any other code, such as 1 for a
 * dataset directory that does not exist, has `category === null`. The message of the engine's
 * error is in the output of the process on the server, which the API does not return.
 */
export class JobFailed extends RemoteError {
  readonly jobId: string;
  readonly jobKind: string;
  readonly exitCode: number | null;
  readonly category: ErrorCategory | null;

  constructor(jobId: string, jobKind: string, exitCode: number | null, message: string) {
    super(null, `${jobKind} job ${jobId}: ${message}`);
    this.jobId = jobId;
    this.jobKind = jobKind;
    this.exitCode = exitCode;
    this.category = exitCategory(exitCode);
  }
}

/** `waitJob` waited `timeout` milliseconds, and the job had not finished. The job keeps running. */
export class JobTimeout extends RemoteError {
  readonly jobId: string;
  readonly timeout: number;

  constructor(jobId: string, timeout: number) {
    super(null, `job ${jobId} did not finish in ${timeout} ms; it is still on the server`);
    this.jobId = jobId;
    this.timeout = timeout;
  }
}

/** The category of engine error that an exit code of the server's command line names. */
export function exitCategory(code: number | null | undefined): ErrorCategory | null {
  switch (code) {
    case 2:
      return InvalidRequest;
    case 4:
      return NotFound;
    case 5:
      return ConflictingState;
    case 70:
      return StorageFailure;
    default:
      return null;
  }
}

type ErrorClass = new (status: number | null, message: string, kind?: string | null) => RemoteError;

/**
 * The classes that the `kind` field of a response can name. The light suite of the server checks
 * that this list has every error of the engine, in the same category.
 */
export const ERROR_KINDS: Readonly<Record<string, ErrorClass>> = Object.freeze({
  PayloadMismatch,
  WrongDimension,
  UnknownBackend,
  InvalidOption,
  UnsupportedOperation,
  ProfileNotInstalled,
  UnknownPartition,
  UnknownFollower,
  PluginNotLoaded,
  PendingBacklog,
  NothingStaged,
  NoIndex,
  NotTrained,
  EmptyProject,
  ProjectClosed,
  StaleFollower,
  HasFollowers,
  CorruptedStorage,
  Unauthenticated,
  Forbidden,
});

/**
 * The error for a response with `status` and the text `body`. The `kind` field of the body
 * selects the class. Without a known `kind`, the status selects the generic class of its
 * category.
 */
export function responseError(status: number, body: string): RemoteError {
  let message = body;
  let kind: string | null = null;
  try {
    const parsed: unknown = JSON.parse(body);
    if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
      const fields = parsed as Record<string, unknown>;
      if (fields.error !== undefined) message = String(fields.error);
      if (typeof fields.kind === "string") kind = fields.kind;
    }
  } catch {
    // A body that is not JSON, such as the page of a proxy, is the message as it is.
  }
  const known = kind === null ? undefined : ERROR_KINDS[kind];
  if (known !== undefined) return new known(status, message, kind);
  switch (status) {
    case 400:
      return new RequestRejected(status, message, kind);
    case 401:
      return new Unauthenticated(status, message, kind);
    case 403:
      return new Forbidden(status, message, kind);
    case 404:
      return new ResourceNotFound(status, message, kind);
    case 409:
      return new StateConflict(status, message, kind);
    case 500:
      return new ServerFailure(status, message, kind);
    default:
      return new UnexpectedResponse(status, message, kind);
  }
}
