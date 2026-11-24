// The token of the client against a server that requires one (PLAN.md §5.17, phase 10b). `api/21`
// of the server's full suite enables the authentication of the shared test server, creates the
// two tokens, and runs this file with SSE_TEST_URL, SSE_TEST_ADMIN_TOKEN and SSE_TEST_READER_TOKEN.
// Without them, the test is skipped.

import { test } from "node:test";
import assert from "node:assert/strict";

import { AccessDenied, Forbidden, SimilaritySearchClient, Unauthenticated } from "../src/index.js";

const url = process.env.SSE_TEST_URL;
const admin = process.env.SSE_TEST_ADMIN_TOKEN;
const reader = process.env.SSE_TEST_READER_TOKEN;
const skip =
  url === undefined || admin === undefined || reader === undefined
    ? "SSE_TEST_URL, SSE_TEST_ADMIN_TOKEN and SSE_TEST_READER_TOKEN are not set: this test needs a server with authentication"
    : false;

test("a token is sent with every request", { skip }, async () => {
  const anonymous = new SimilaritySearchClient(url);
  // The routes of control answer without a token.
  assert.equal((await anonymous.healthz()).status, "ok");
  const e1 = await anonymous.listDatasets().then(() => null, (e: unknown) => e);
  assert.ok(e1 instanceof Unauthenticated && e1 instanceof AccessDenied);
  assert.equal(e1.status, 401);

  assert.ok(Array.isArray((await new SimilaritySearchClient(url, { token: admin }).listDatasets()).datasets));

  const asReader = new SimilaritySearchClient(url, { token: reader });
  assert.ok(Array.isArray((await asReader.listDatasets()).datasets));
  const e2 = await asReader.deleteDataset("nc_any").then(() => null, (e: unknown) => e);
  assert.ok(e2 instanceof Forbidden);
  assert.equal(e2.status, 403);
  assert.match(e2.message, /admin:nc_any/);
});
