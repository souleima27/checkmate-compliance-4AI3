"use client";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import ThemeToggler from "./ThemeToggler";
import menuData from "./menuData";

const Header = () => {
  const [navbarOpen, setNavbarOpen] = useState(false);
  const navbarToggleHandler = () => setNavbarOpen(!navbarOpen);

  const [sticky, setSticky] = useState(false);
  const handleStickyNavbar = () => {
    setSticky(window.scrollY >= 80);
  };
  useEffect(() => {
    window.addEventListener("scroll", handleStickyNavbar);
  });

  const [openIndex, setOpenIndex] = useState(-1);
  const handleSubmenu = (index: number) => {
    setOpenIndex(openIndex === index ? -1 : index);
  };

  const usePathName = usePathname();

  return (
    <header
      className={`header top-0 left-0 z-40 flex w-full items-center ${
        sticky
          ? "dark:bg-gray-dark dark:shadow-sticky-dark shadow-sticky fixed z-9999 bg-white/80 backdrop-blur-xs transition"
          : "absolute bg-transparent"
      }`}
    >
      <div className="container">
        <div className="relative -mx-4 flex items-center justify-between">
          {/* Logo */}
          <div className="w-60 max-w-full px-4 xl:mr-12">
            <Link
              href="/"
              className={`header-logo block w-full ${
                sticky ? "py-5 lg:py-2" : "py-8"
              } `}
            >
              <Image
                src="/images/logo/logo-2.svg"
                alt="CheckMate logo"
                width={140}
                height={30}
                className="w-full dark:hidden"
              />
              <Image
                src="/images/logo/logo.svg"
                alt="CheckMate logo"
                width={140}
                height={30}
                className="hidden w-full dark:block"
              />
            </Link>
          </div>

          {/* Navigation */}
          <div className="flex w-full items-center justify-between px-4">
            <div>
              {/* Mobile menu toggle */}
              <button
                onClick={navbarToggleHandler}
                id="navbarToggler"
                aria-label="Menu mobile"
                className="ring-primary absolute top-1/2 right-4 block translate-y-[-50%] rounded-lg px-3 py-[6px] focus:ring-2 lg:hidden"
              >
                <span
                  className={`relative my-1.5 block h-0.5 w-[30px] bg-black transition-all duration-300 dark:bg-white ${
                    navbarOpen ? "top-[7px] rotate-45" : ""
                  }`}
                />
                <span
                  className={`relative my-1.5 block h-0.5 w-[30px] bg-black transition-all duration-300 dark:bg-white ${
                    navbarOpen ? "opacity-0" : ""
                  }`}
                />
                <span
                  className={`relative my-1.5 block h-0.5 w-[30px] bg-black transition-all duration-300 dark:bg-white ${
                    navbarOpen ? "top-[-8px] -rotate-45" : ""
                  }`}
                />
              </button>

              {/* Menu items */}
              <nav
                id="navbarCollapse"
                className={`navbar absolute right-0 z-30 w-[250px] rounded border bg-white px-6 py-4 duration-300 dark:bg-dark lg:visible lg:static lg:w-auto lg:border-none lg:bg-transparent lg:p-0 lg:opacity-100 ${
                  navbarOpen
                    ? "visibility top-full opacity-100"
                    : "invisible top-[120%] opacity-0"
                }`}
              >
                <ul className="block lg:flex lg:space-x-12">
                  {menuData.map((menuItem, index) => (
                    <li key={index} className="group relative">
                      <Link
                        href={menuItem.path}
                        className={`flex py-2 text-base lg:inline-flex lg:px-0 lg:py-6 ${
                          usePathName === menuItem.path
                            ? "text-primary dark:text-white"
                            : "text-dark hover:text-primary dark:text-white/70 dark:hover:text-white"
                        }`}
                      >
                        {menuItem.title}
                      </Link>
                    </li>
                  ))}
                </ul>
              </nav>
            </div>

            {/* Actions */}
            <div className="flex items-center justify-end pr-16 lg:pr-0">
              <div>
                <ThemeToggler />
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
