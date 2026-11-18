// Stores the print file for a print/gift order, so it can be sent to the print-on-demand service.
// Called only when someone orders. Returns an id that is passed to Stripe as client_reference_id:
// in the Stripe order you'll see that id, and the file is at prints/<id>.jpg in your Vercel Blob store.
// Env var: BLOB_READ_WRITE_TOKEN (created automatically when you add a Blob store to the project).
import { put } from "@vercel/blob";
import crypto from "node:crypto";

const PRODUCTS = new Set(["poster-a3", "canvas-30", "couple-poster", "phone-case", "pendant", "mug"]);
const MAX = 8 * 1024 * 1024;

async function readBody(req) {
  const chunks = [];
  let n = 0;
  for await (const c of req) {
    n += c.length;
    if (n > MAX) throw new Error("too big");
    chunks.push(c);
  }
  return Buffer.concat(chunks);
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();
  const product = String((req.query && req.query.product) || "");
  if (!PRODUCTS.has(product)) return res.status(400).json({ error: "unknown product" });
  let body;
  try { body = await readBody(req); } catch (e) { return res.status(413).json({ error: "too big" }); }
  if (body.length < 1000 || body[0] !== 0xff || body[1] !== 0xd8) return res.status(400).json({ error: "not a jpeg" });
  const id = crypto.randomBytes(9).toString("hex");
  const meta = JSON.stringify({ product, created: new Date().toISOString() });
  // private store if you have one (files only visible from your Vercel dashboard), else public with an unguessable name
  for (const access of ["private", "public"]) {
    try {
      await put(`prints/${id}.jpg`, body, { access, contentType: "image/jpeg", addRandomSuffix: false });
      await put(`prints/${id}.json`, meta, { access, contentType: "application/json", addRandomSuffix: false });
      return res.status(200).json({ id });
    } catch (e) { /* try the next mode */ }
  }
  return res.status(500).json({ error: "storage not configured" });
}
