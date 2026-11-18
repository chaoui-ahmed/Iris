// Confirms a Stripe Checkout payment for the HD pack, so the unlock can't be faked by typing a URL.
// Env vars (Vercel → Project → Settings → Environment Variables):
//   STRIPE_SECRET_KEY   a restricted key with read access to Checkout Sessions (rk_live_... or sk_...)
//   STRIPE_HD_LINK_ID   optional: the HD Payment Link id (plink_...), so other payments don't unlock HD
export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  const id = String((req.query && req.query.session_id) || "");
  if (!/^cs_(test|live)_[A-Za-z0-9]+$/.test(id)) return res.status(400).json({ paid: false });
  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) return res.status(500).json({ paid: false, error: "not configured" });
  try {
    const r = await fetch("https://api.stripe.com/v1/checkout/sessions/" + id, { headers: { Authorization: "Bearer " + key } });
    if (!r.ok) return res.status(404).json({ paid: false });
    const s = await r.json();
    const link = process.env.STRIPE_HD_LINK_ID;
    const paid = s.payment_status === "paid" && (!link || s.payment_link === link);
    return res.status(200).json({ paid });
  } catch (e) {
    return res.status(502).json({ paid: false });
  }
}
