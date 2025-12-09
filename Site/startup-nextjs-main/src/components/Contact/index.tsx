import NewsLatterBox from "./NewsLatterBox";

const Contact = () => {
  return (
    <section id="contact" className="overflow-hidden py-16 md:py-20 lg:py-28 bg-[var(--color-page-bg)]">
      <div className="container">
        <div className="-mx-4 flex flex-wrap">
          {/* Formulaire de contact */}
          <div className="w-full px-4 lg:w-7/12 xl:w-8/12">
            <div
              className="mb-12 rounded-xs bg-white px-8 py-11 shadow-three dark:bg-gray-dark sm:p-[55px] lg:mb-5 lg:px-8 xl:p-[55px]"
              data-wow-delay=".15s"
            >
              <h2 className="mb-3 text-2xl font-bold text-black dark:text-white sm:text-3xl lg:text-2xl xl:text-3xl">
                Besoin d’aide ? Ouvrez un ticket
              </h2>
              <p className="mb-12 text-base font-medium text-black">
                Notre équipe de support vous répondra rapidement par email.
              </p>
              <form>
                <div className="-mx-4 flex flex-wrap">
                  {/* Nom */}
                  <div className="w-full px-4 md:w-1/2">
                    <div className="mb-8">
                      <label
                        htmlFor="name"
                        className="mb-3 block text-sm font-medium text-dark dark:text-white"
                      >
                        Votre nom
                      </label>
                      <input
                        type="text"
                        placeholder="Entrez votre nom"
                        className="border-stroke w-full rounded-xs border bg-[#f8f8f8] px-6 py-3 text-base text-black outline-hidden focus:border-primary dark:border-transparent dark:bg-[#2C303B] dark:text-white dark:shadow-two dark:focus:border-primary dark:focus:shadow-none"
                      />
                    </div>
                  </div>
                  {/* Email */}
                  <div className="w-full px-4 md:w-1/2">
                    <div className="mb-8">
                      <label
                        htmlFor="email"
                        className="mb-3 block text-sm font-medium text-dark dark:text-white"
                      >
                        Votre email
                      </label>
                      <input
                        type="email"
                        placeholder="Entrez votre email"
                        className="border-stroke w-full rounded-xs border bg-[#f8f8f8] px-6 py-3 text-base text-black outline-hidden focus:border-primary dark:border-transparent dark:bg-[#2C303B] dark:text-white dark:shadow-two dark:focus:border-primary dark:focus:shadow-none"
                      />
                    </div>
                  </div>
                  {/* Message */}
                  <div className="w-full px-4">
                    <div className="mb-8">
                      <label
                        htmlFor="message"
                        className="mb-3 block text-sm font-medium text-dark dark:text-white"
                      >
                        Votre message
                      </label>
                      <textarea
                        name="message"
                        rows={5}
                        placeholder="Entrez votre message"
                        className="border-stroke w-full resize-none rounded-xs border bg-[#f8f8f8] px-6 py-3 text-base text-black outline-hidden focus:border-primary dark:border-transparent dark:bg-[#2C303B] dark:text-white dark:shadow-two dark:focus:border-primary dark:focus:shadow-none"
                      ></textarea>
                    </div>
                  </div>
                  {/* Bouton */}
                  <div className="w-full px-4">
                    <button className="rounded-xs bg-primary px-9 py-4 text-base font-medium text-white shadow-submit duration-300 hover:bg-primary/90 dark:shadow-submit-dark">
                      Envoyer le ticket
                    </button>
                  </div>
                </div>
              </form>
            </div>
          </div>

          {/* Newsletter / info complémentaire */}
          <div className="w-full px-4 lg:w-5/12 xl:w-4/12">
            <NewsLatterBox />
          </div>
        </div>
      </div>
    </section>
  );
};

export default Contact;
