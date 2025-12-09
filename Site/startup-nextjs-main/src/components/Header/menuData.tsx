import { Menu } from "@/types/menu";

const menuData: Menu[] = [
  {
    id: 1,
    title: "Accueil",
    path: "/",
    newTab: false,
  },
  {
    id: 2,
    title: "À propos",
    path: "/about",
    newTab: false,
  },

  {
    id: 3,
    title: "Importer",
    path: "/upload",
    newTab: false,
  },
  {
    id: 4,
    title: "Analyse",
    path: "/review",
    newTab: false,
  },
  {
    id: 5,
    title: "Tableau de bord",
    path: "/dashboard",
    newTab: false,
  },
];
export default menuData;
