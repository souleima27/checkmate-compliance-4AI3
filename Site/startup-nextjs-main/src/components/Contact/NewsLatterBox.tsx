"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";



const NewsLatterBox = () => {
  const { theme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  // Decide color safely
  const gradientColor = mounted
    ? theme === "light"
      ? "#19A974"
      : "#fff"
      : "#fff";

  return (
    <div className="shadow-three dark:bg-gray-dark relative z-10 rounded-xs bg-white p-8 sm:p-11 lg:p-8 xl:p-11">
      <h3 className="mb-4 text-2xl leading-tight font-bold text-black dark:text-white">
        Abonnez-vous pour recevoir nos mises à jour
      </h3>
      <p className="border-body-color/25 text-black mb-11 border-b pb-11 text-base leading-relaxed dark:text-white dark:border-white/25">
        Restez informé des dernières fonctionnalités de CheckMate et des
        nouveautés en matière de conformité.
      </p>
      <div>
        <input
          type="text"
          name="name"
          placeholder="Entrez votre nom"
          className="border-stroke text-black focus:border-primary dark:text-white dark:shadow-two dark:focus:border-primary mb-4 w-full rounded-xs border bg-[#f8f8f8] px-6 py-3 text-base outline-hidden dark:border-transparent dark:bg-[#2C303B] dark:focus:shadow-none"
        />
        <input
          type="email"
          name="email"
          placeholder="Entrez votre email"
          className="border-stroke text-black focus:border-primary dark:text-white dark:shadow-two dark:focus:border-primary mb-4 w-full rounded-xs border bg-[#f8f8f8] px-6 py-3 text-base outline-hidden dark:border-transparent dark:bg-[#2C303B] dark:focus:shadow-none"
        />
        <input
          type="submit"
          value="S’abonner"
          className="bg-primary shadow-submit hover:bg-primary/90 dark:shadow-submit-dark mb-5 flex w-full cursor-pointer items-center justify-center rounded-xs px-9 py-4 text-base font-medium text-white duration-300"
        />
        <p className="text-black dark:text-white text-center text-base leading-relaxed">
          Garantie sans spam — uniquement des informations utiles et pertinentes.
        </p>
      </div>

      {/* Décorations SVG conservées avec palette verte */}
      <div>
      <span className="absolute top-7 left-2">
        <svg
          width="57"
          height="65"
          viewBox="0 0 57 65"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            opacity="0.5"
            d="M0.407629 15.9573L39.1541 64.0714L56.4489 0.160793L0.407629 15.9573Z"
            fill="url(#paint0_linear_1028_600)"
          />
          <defs>
            <linearGradient
              id="paint0_linear_1028_600"
              x1="-18.3187"
              y1="55.1044"
              x2="37.161"
              y2="15.3509"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor={gradientColor} stopOpacity="0.62" />
              <stop offset="1" stopColor={gradientColor} stopOpacity="0" />
            </linearGradient>
          </defs>
        </svg>
      </span>
        {/* autres décorations SVG inchangées */}
      </div>
    </div>
  );
};

export default NewsLatterBox;
