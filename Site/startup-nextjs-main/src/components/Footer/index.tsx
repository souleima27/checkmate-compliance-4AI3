"use client";
import Image from "next/image";
import Link from "next/link";

const Footer = () => {
  return (
    <footer className="relative z-10 bg-white pt-16 md:pt-20 lg:pt-24">
      <div className="container">
        <div className="-mx-4 flex flex-wrap">
          {/* Logo + description */}
          <div className="w-full px-4 md:w-1/2 lg:w-4/12 xl:w-5/12">
            <div className="mb-12 max-w-[360px] lg:mb-16">
              <Link href="/" className="mb-8 inline-block">
                <Image
                  src="/images/logo/logo.svg"
                  alt="CheckMate logo"
                  width={160}
                  height={40}
                  className="w-full"
                />
              </Link>
              <p className="mb-9 text-base leading-relaxed text-black">
                CheckMate vous aide à analyser, vérifier et harmoniser vos
                documents marketing selon les règles de conformité.
              </p>
              
            </div>
          </div>

          {/* Liens de navigation */}
          <div className="w-full px-4 sm:w-1/2 md:w-1/2 lg:w-2/12 xl:w-2/12">
            <div className="mb-12 lg:mb-16">
              <h2 className="mb-10 text-xl font-bold text-black">Navigation</h2>
              <ul>
                <li>
                  <Link href="/about" className="mb-4 inline-block text-base text-black hover:text-[var(--color-primary)]">
                    À propos
                  </Link>
                </li>
                <li>
                  <Link href="/upload" className="mb-4 inline-block text-base text-black hover:text-[var(--color-primary)]">
                    Importer
                  </Link>
                </li>
                <li>
                  <Link href="/review" className="mb-4 inline-block text-base text-black hover:text-[var(--color-primary)]">
                    Revue
                  </Link>
                </li>
                <li>
                  <Link href="/dashboard" className="mb-4 inline-block text-base text-black hover:text-[var(--color-primary)]">
                    Tableau de bord
                  </Link>
                </li>
              </ul>
            </div>
          </div>

          {/* Conditions */}
          <div className="w-full px-4 sm:w-1/2 md:w-1/2 lg:w-2/12 xl:w-2/12">
            <div className="mb-12 lg:mb-16">
              <h2 className="mb-10 text-xl font-bold text-black">Conditions</h2>
              <ul>
                <li>
                  <Link href="/terms" className="mb-4 inline-block text-base text-black hover:text-[var(--color-primary)]">
                    Conditions d’utilisation
                  </Link>
                </li>
                <li>
                  <Link href="/privacy" className="mb-4 inline-block text-base text-black hover:text-[var(--color-primary)]">
                    Politique de confidentialité
                  </Link>
                </li>
              </ul>
            </div>
          </div>

          {/* Support */}
          <div className="w-full px-4 md:w-1/2 lg:w-4/12 xl:w-3/12">
            <div className="mb-12 lg:mb-16">
              <h2 className="mb-10 text-xl font-bold text-black">Support & Aide</h2>
              <ul>
                <li>
                  <Link href="/contact" className="mb-4 inline-block text-base text-black hover:text-[var(--color-primary)]">
                    Ouvrir un ticket
                  </Link>
                </li>
                <li>
                  <Link href="/error" className="mb-4 inline-block text-base text-black hover:text-[var(--color-primary)]">
                    Page d’erreur
                  </Link>
                </li>
                <li>
                  <Link href="/signin" className="mb-4 inline-block text-base text-black hover:text-[var(--color-primary)]">
                    Connexion
                  </Link>
                </li>
                <li>
                  <Link href="/signup" className="mb-4 inline-block text-base text-black hover:text-[var(--color-primary)]">
                    Inscription
                  </Link>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Bas de page */}
        <div className="h-px w-full bg-gray-200"></div>
        <div className="py-8">
          <p className="text-center text-base text-black">
            © {new Date().getFullYear()} CheckMate — Tous droits réservés.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
