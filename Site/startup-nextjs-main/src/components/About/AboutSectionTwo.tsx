import Image from "next/image";

const AboutSectionTwo = () => {
  return (
    <section className="py-16 md:py-20 lg:py-28 bg-[var(--color-page-bg)]">
      <div className="container">
        <div className="-mx-4 flex flex-wrap items-center">
          <div className="w-full px-4 lg:w-1/2">
            <div className="relative mx-auto mb-12 aspect-25/24 max-w-[500px] text-center lg:m-0" data-wow-delay=".15s">
              <Image
                src="/images/about/about-image-2.svg"
                alt="illustration analyse"
                fill
                className="drop-shadow-three dark:hidden"
              />
              <Image
                src="/images/about/about-image-2-dark.svg"
                alt="illustration analyse"
                fill
                className="hidden drop-shadow-three dark:block"
              />
            </div>
          </div>

          <div className="w-full px-4 lg:w-1/2">
            <div className="max-w-[470px]">
              <div className="mb-9">
                <h3 className="mb-4 text-xl font-bold text-black sm:text-2xl lg:text-xl xl:text-2xl">
                  Analyse fiable et sans erreur
                </h3>
                <p className="text-base font-medium leading-relaxed text-black sm:text-lg">
                  CheckMate détecte les incohérences structurelles et contextuelles pour garantir la qualité de vos documents.
                </p>
              </div>
              <div className="mb-9">
                <h3 className="mb-4 text-xl font-bold text-black sm:text-2xl lg:text-xl xl:text-2xl">
                  Assistance dédiée
                </h3>
                <p className="text-base font-medium leading-relaxed text-black sm:text-lg">
                  Notre équipe vous accompagne dans la configuration, l’analyse et la validation de vos contenus.
                </p>
              </div>
              <div className="mb-1">
                <h3 className="mb-4 text-xl font-bold text-black sm:text-2xl lg:text-xl xl:text-2xl">
                  Compatible Next.js & Tailwind
                </h3>
                <p className="text-base font-medium leading-relaxed text-black sm:text-lg">
                  CheckMate est construit avec les dernières technologies pour une intégration fluide et rapide.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutSectionTwo;
