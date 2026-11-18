/* IRIS settings. Fill these in to switch on payments, prints and ads (see MONETIZATION.md).
   Anything left empty stays hidden in the app. Nothing secret goes in here: this file is public. */
window.IRIS_CONFIG = {
  siteUrl: "https://iriseye.vercel.app",

  // Paid HD pack: a Stripe Payment Link. In Stripe, set its success page to
  //   https://iriseye.vercel.app/?unlock=hd&session_id={CHECKOUT_SESSION_ID}
  hd: { paymentLink: "", price: "€2.99" },

  // Prints and gifts: one Stripe Payment Link per product (collect the shipping address in Stripe).
  // Success page for all of them: https://iriseye.vercel.app/?order=done
  prints: [
    { id: "poster-a3", name: "Poster A3", price: "€29", link: "" },
    { id: "canvas-30", name: "Canvas 30 × 30 cm", price: "€49", link: "" },
    { id: "couple-poster", name: "Couple poster (fusion)", price: "€39", link: "" },
    { id: "phone-case", name: "Phone case", price: "€25", link: "" },
    { id: "pendant", name: "Necklace pendant", price: "€35", link: "" },
    { id: "mug", name: "Mug", price: "€19", link: "" }
  ],

  // Google AdSense: shown only on the share and order-confirmation screens, never to HD buyers.
  adsense: { client: "", slot: "" },

  contact: "https://github.com/chaoui-ahmed/iris/issues"
};
