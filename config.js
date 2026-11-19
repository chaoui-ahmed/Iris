/* IRIS settings. Payment links below are Stripe LIVE mode.
   IRIS settings. Fill these in to switch on payments, prints and ads (see MONETIZATION.md).
   Anything left empty stays hidden in the app. Nothing secret goes in here: this file is public. */
window.IRIS_CONFIG = {
  siteUrl: "https://iriseye.vercel.app",

  // Paid HD pack: a Stripe Payment Link. In Stripe, set its success page to
  //   https://iriseye.vercel.app/?unlock=hd&session_id={CHECKOUT_SESSION_ID}
  hd: { paymentLink: "https://buy.stripe.com/aFa6oG1HUd3uaaKfzBaMU06", price: "€2.99" },

  // Prints and gifts: one Stripe Payment Link per product (collect the shipping address in Stripe).
  // Success page for all of them: https://iriseye.vercel.app/?order=done
  prints: [
    { id: "poster-a3", name: "Poster A3", price: "€29", link: "https://buy.stripe.com/5kQ4gy2LYbZq1EegDFaMU05" },
    { id: "canvas-30", name: "Canvas 30 × 30 cm", price: "€49", link: "https://buy.stripe.com/28E5kC1HU3sUfv45Z1aMU04" },
    { id: "couple-poster", name: "Couple poster (fusion)", price: "€39", link: "https://buy.stripe.com/4gMdR886i4wY6Yy0EHaMU03" },
    { id: "phone-case", name: "Phone case", price: "€25", link: "https://buy.stripe.com/cNi8wOcmye7yfv4gDFaMU02" },
    { id: "pendant", name: "Necklace pendant", price: "€35", link: "https://buy.stripe.com/7sY7sKbiufbC3MmevxaMU01" },
    { id: "mug", name: "Mug", price: "€19", link: "https://buy.stripe.com/9B6eVc2LYfbCer0afhaMU00" }
  ],

  // Google AdSense: shown only on the share and order-confirmation screens, never to HD buyers.
  adsense: { client: "", slot: "" },

  contact: "https://github.com/chaoui-ahmed/iris/issues"
};
