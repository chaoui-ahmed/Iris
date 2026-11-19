/* IRIS settings. Links below are Stripe TEST mode (buy.stripe.com/test_...): swap them for live links before launch.
   IRIS settings. Fill these in to switch on payments, prints and ads (see MONETIZATION.md).
   Anything left empty stays hidden in the app. Nothing secret goes in here: this file is public. */
window.IRIS_CONFIG = {
  siteUrl: "https://iriseye.vercel.app",

  // Paid HD pack: a Stripe Payment Link. In Stripe, set its success page to
  //   https://iriseye.vercel.app/?unlock=hd&session_id={CHECKOUT_SESSION_ID}
  hd: { paymentLink: "https://buy.stripe.com/test_14A28q0LA8X87lp86A97G00", price: "€2.99" },

  // Prints and gifts: one Stripe Payment Link per product (collect the shipping address in Stripe).
  // Success page for all of them: https://iriseye.vercel.app/?order=done
  prints: [
    { id: "poster-a3", name: "Poster A3", price: "€29", link: "https://buy.stripe.com/test_5kQ3cuama0qC35972w97G01" },
    { id: "canvas-30", name: "Canvas 30 × 30 cm", price: "€49", link: "https://buy.stripe.com/test_aFa28qeCqa1cfRV86A97G02" },
    { id: "couple-poster", name: "Couple poster (fusion)", price: "€39", link: "https://buy.stripe.com/test_14A8wOcui4GS8ptbiM97G03" },
    { id: "phone-case", name: "Phone case", price: "€25", link: "https://buy.stripe.com/test_28E14m8e2flw3591Ic97G04" },
    { id: "pendant", name: "Necklace pendant", price: "€35", link: "https://buy.stripe.com/test_00wcN49i62yKeNR1Ic97G05" },
    { id: "mug", name: "Mug", price: "€19", link: "https://buy.stripe.com/test_4gMdR865U5KWdJN86A97G06" }
  ],

  // Google AdSense: shown only on the share and order-confirmation screens, never to HD buyers.
  adsense: { client: "", slot: "" },

  contact: "https://github.com/chaoui-ahmed/iris/issues"
};
