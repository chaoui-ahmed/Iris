# Making money with IRIS: setup

Everything is built in. Each feature stays hidden until you fill in its part of `config.js`, so you can switch them on one at a time.

| Feature | What the user sees | You need |
|---|---|---|
| HD pack | "Get HD, no watermark · €2.99" on the share screen. Unlocks a 4K export and clean cards/video on that phone. | Stripe account, 1 Payment Link, 2 env vars |
| Prints and gifts | "Order a print or gift" → product list → Stripe checkout with shipping | 1 Payment Link per product, a Vercel Blob store, a print-on-demand account |
| Ads | One ad under the share buttons and on the order confirmation. Never during the scan, never for HD buyers. | Google AdSense account (approval takes days to weeks) |
| Growth | Site address on every free card and video, "Send IRIS to a friend", link previews | Nothing, already on |
| Privacy | `privacy.html`, linked from the home screen | Nothing. Update it if you change how data is handled. |

Free tiers cover all of it: Vercel Hobby (hosting, functions, Blob), Stripe (fees per sale only), print-on-demand (you pay per order, after the customer pays you).

## 1. HD pack (Stripe)

1. Create a Stripe account (stripe.com), finish the business details so you can receive payouts. As a student in France you can start as an individual / micro-entrepreneur.
2. Products → Add product: "IRIS HD pack", one-off price €2.99.
3. Payment Links → New → that product. Under "After payment", choose "Don't show confirmation page" and set the redirect URL to:
   `https://iriseye.vercel.app/?unlock=hd&session_id={CHECKOUT_SESSION_ID}`
   (type `{CHECKOUT_SESSION_ID}` exactly like that; Stripe fills it in.)
4. Copy the link (`https://buy.stripe.com/...`) into `config.js` → `hd.paymentLink`.
5. Developers → API keys → Create restricted key with "Checkout Sessions: Read" only. In Vercel → Project → Settings → Environment Variables add:
   - `STRIPE_SECRET_KEY` = that restricted key
   - `STRIPE_HD_LINK_ID` = the Payment Link's id (`plink_...`, shown on the link's page). Optional, but stops other payments from unlocking HD.
6. Redeploy. Test first with Stripe test mode (test link + test key, card `4242 4242 4242 4242`).

How it works: before leaving for Stripe, the app saves the current scan on the phone (IndexedDB, never uploaded). Stripe sends the user back with a session id, `/api/verify` asks Stripe whether that session was paid, and the app restores the scan with HD unlocked.

Known limit: the unlock is stored in the browser, so it's per phone/browser, and clearing site data removes it. Fine for €2.99; accounts would be the next step if it sells.

## 2. Prints and gifts

The flow: customer picks a product → the app renders the 4K picture and uploads it (`/api/upload` → Vercel Blob, `prints/<id>.jpg`) → Stripe checkout with `client_reference_id=<id>` → you see the paid order in Stripe with that id → you send the file to your print-on-demand partner.

1. Vercel → Project → Storage → Create → Blob. Connect it to the project: `BLOB_READ_WRITE_TOKEN` is added for you. Redeploy.
2. Print-on-demand: open an account on Gelato (good EU coverage, local printing) or Printful. Check the base price + shipping of each product, then set your price (aim for 2–3x the base cost).
3. In Stripe, one product + Payment Link per item (prices from `config.js` or your own). For each link:
   - enable "Collect customers' addresses" → shipping address, and the countries you ship to
   - "After payment" redirect: `https://iriseye.vercel.app/?order=done`
4. Paste each link in `config.js` → `prints[].link`. Products with an empty link stay hidden. To add/remove products, edit both `config.js` and the `PRODUCTS` list in `api/upload.js`.
5. Fulfilling an order (manual to start, ~2 minutes each):
   - Stripe → Payments → open the order → note the `client_reference_id` and the shipping address
   - Vercel → Storage → your Blob store → `prints/<id>.jpg` → download
   - Gelato/Printful → new order → upload the file → customer's address → pay the base cost
   - When it sells regularly, automate this with a Stripe webhook that calls the Gelato/Printful API.
6. Delete print files once orders have shipped (privacy policy promises it).

Prints that should sell best: the couple fusion poster (Valentine's, anniversaries, weddings) and phone cases.

## 3. Ads (Google AdSense)

1. Apply at adsense.google.com with `iriseye.vercel.app` (Vercel subdomains can be refused: a custom domain, around €10/year, helps a lot).
2. Once approved: create a display ad unit, copy the publisher id (`ca-pub-...`) and the slot id into `config.js` → `adsense`.
3. Add a file `ads.txt` at the root of the repo with the line AdSense gives you.
4. In AdSense → Privacy & messaging, turn on the GDPR consent message (required for visitors in the EU/UK).

Honest expectation: ads earn little per visit. Keep them as a small extra; HD and prints are the real revenue.

## 4. Before charging money

- **Iris textures licence.** The renderer uses textures from the Synthetic Iris Image Dataset (Zhang et al., 2025). Hugging Face says MIT, the authors' README says research use. Email the authors (contact in their README) for written permission, or rebuild `lib/iris/` from your own macro shots or licensed images (`training/build_library.py` shows the format).
- **Legal basics in France.** Selling online needs a status (micro-entreprise is the simple one), mentions légales on the site (who runs it, contact) and terms of sale (CGV: prices, delivery, 14-day withdrawal right, which doesn't apply to personalised products like prints, but say so).
- **Test on real eyes** before spending on promotion.

## 5. Growth checklist

- Post collision videos on TikTok / Instagram Reels from your own account to seed it.
- Every free card and video carries `iriseye.vercel.app`. A short custom domain is easier to remember and type.
- "Send IRIS to a friend" (on the fusion screen) shares a link that opens the app with an invite line.
- Link previews (WhatsApp, iMessage, X) use `og.jpg`.
