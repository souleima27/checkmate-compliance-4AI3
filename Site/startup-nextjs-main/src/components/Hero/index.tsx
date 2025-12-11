import Link from "next/link";

const Hero = () => {
  return (
    <section
      id="accueil"
      className="relative z-10 overflow-hidden bg-white pb-16 pt-[120px] md:pb-[120px] md:pt-[150px] xl:pb-[160px] xl:pt-[180px] 2xl:pb-[200px] 2xl:pt-[210px]"
    >
      <div className="container">
        <div className="-mx-4 flex flex-wrap">
          <div className="w-full px-4">
            <div className="mx-auto max-w-[900px] text-center">
              {/* Titre principal */}
              <h1 className="mb-6 text-3xl font-bold leading-tight text-black sm:text-4xl md:text-5xl">
                Bienvenue sur <span className="text-[var(--color-primary)]">CheckMate</span>
              </h1>
              <h2 className="mb-6 text-xl font-semibold text-black sm:text-2xl md:text-3xl">
                L’IA qui sécurise vos documents marketing
              </h2>

              {/* Slogan */}
              <p className="mb-12 text-base leading-relaxed text-black sm:text-lg md:text-xl">
                Vérifiez vos présentations et prospectus en un clic.  
                Détection automatique des violations, annotations claires, rapport final prêt à l’audit.
              </p>

              {/* Boutons d’action */}
              <div className="flex flex-col items-center justify-center space-y-4 sm:flex-row sm:space-x-4 sm:space-y-0">
                <Link
                  href="/upload"
                  aria-label="Uploader présentation et prospectus"
                  className="rounded-xs bg-[var(--color-primary)] px-8 py-4 text-base font-semibold text-white duration-300 ease-in-out hover:bg-[var(--color-primary-dark)]"
                >
                  Importer Documents Marketing
                </Link>
                <Link
                  href="/review"
                  aria-label="Accéder à l’analyse"
                  className="rounded-xs bg-[var(--color-light-green)] px-8 py-4 text-base font-semibold text-black duration-300 ease-in-out hover:bg-[var(--color-light-green-alt)]"
                >
                  Accéder à l’analyse
                </Link>
              </div>

              {/* Cartes explicatives */}
              <div className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                {/* Carte 1 : Règles */}
                <div className="rounded-lg border border-[var(--color-stroke-stroke)] bg-[var(--color-page-bg-alt)] p-6 text-left">
                  <h3 className="mb-2 text-lg font-semibold text-black">Règles de conformité</h3>
                  <p className="mb-4 text-black">
                    Basé sur le Glossaire officiel et la Synthèse des règles fournies.
                  </p>
                  <div className="flex flex-wrap gap-2 text-sm">
                    <span className="inline-flex rounded bg-[var(--color-primary)] px-2 py-1 font-semibold text-white">Structurelles</span>
                    <span className="inline-flex rounded bg-[var(--color-primary)] px-2 py-1 font-semibold text-white">Contextuelles</span>
                    <span className="inline-flex rounded bg-[var(--color-primary)] px-2 py-1 font-semibold text-white">Fonds</span>
                    <span className="inline-flex rounded bg-[var(--color-primary)] px-2 py-1 font-semibold text-white">Disclaimers</span>
                  </div>
                </div>

                {/* Carte 2 : Documents */}
                <div className="rounded-lg border border-[var(--color-stroke-stroke)] bg-[var(--color-page-bg-alt)] p-6 text-left">
                  <h3 className="mb-2 text-lg font-semibold text-black">Documents marketing</h3>
                  <p className="mb-4 text-black">
                    Importez vos présentations et prospectus, suivez leur statut et lancez l’analyse IA en un clic.
                  </p>
                  <div className="flex flex-wrap gap-2 text-sm">
                    <span className="inline-flex rounded bg-[var(--color-light-green)] px-2 py-1 font-semibold text-black">Non analysé</span>
                    <span className="inline-flex rounded bg-[var(--color-light-green)] px-2 py-1 font-semibold text-black">En cours</span>
                    <span className="inline-flex rounded bg-[var(--color-primary)] px-2 py-1 font-semibold text-white">Terminé</span>
                  </div>
                </div>

                {/* Carte 3 : Rapport final */}
                <div className="rounded-lg border border-[var(--color-stroke-stroke)] bg-[var(--color-page-bg-alt)] p-6 text-left">
                  <h3 className="mb-2 text-lg font-semibold text-black">Rapport final</h3>
                  <p className="mb-4 text-black">
                    Obtenez un rapport clair avec le nombre de violations par catégorie et un document annoté prêt à l’audit.
                  </p>
                  <div className="flex flex-wrap gap-2 text-sm">
                    <span className="inline-flex rounded bg-red-500 px-2 py-1 font-semibold text-white">Graves</span>
                    <span className="inline-flex rounded bg-yellow-400 px-2 py-1 font-semibold text-black">Moyennes</span>
                    <span className="inline-flex rounded bg-blue-400 px-2 py-1 font-semibold text-white">Mineures</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 🎨 Même background SVG que SigninPage */}
        <div className="absolute top-0 left-0 z-[-1]">
          <svg
            width="1440"
            height="969"
            viewBox="0 0 1440 969"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <mask
              id="mask0_hero"
              style={{ maskType: "alpha" }}
              maskUnits="userSpaceOnUse"
              x="0"
              y="0"
              width="1440"
              height="969"
            >
              <rect width="1440" height="969" fill="#090E34" />
            </mask>
            <g mask="url(#mask0_hero)">
              <path
                opacity="0.1"
                d="M1086.96 297.978L632.959 554.978L935.625 535.926L1086.96 297.978Z"
                fill="url(#paint0_linear_hero)"
              />
              <path
                opacity="0.1"
                d="M1324.5 755.5L1450 687V886.5L1324.5 967.5L-10 288L1324.5 755.5Z"
                fill="url(#paint1_linear_hero)"
              />
            </g>
            <defs>
              <linearGradient
                id="paint0_linear_hero"
                x1="1178.4"
                y1="151.853"
                x2="780.959"
                y2="453.581"
                gradientUnits="userSpaceOnUse"
              >
                <stop stopColor="#bff3ccff" />
                <stop offset="1" stopColor="#bff3ccff" stopOpacity="0" />
              </linearGradient>
              <linearGradient
                id="paint1_linear_hero"
                x1="160.5"
                y1="220"
                x2="1099.45"
                y2="1192.04"
                gradientUnits="userSpaceOnUse"
              >
                <stop stopColor="#19A974" />
                <stop offset="1" stopColor="#19A974" stopOpacity="0" />
              </linearGradient>
            </defs>
          </svg>
        </div>
      </div>
    </section>
  );
};

export default Hero;
