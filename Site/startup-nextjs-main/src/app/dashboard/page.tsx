import Breadcrumb from "@/components/Common/Breadcrumb";
import DashboardTable from "@/components/Dashboard/DashboardTable";
import DashboardStats from "@/components/Dashboard/DashboardStats";
import DashboardTimeline from "@/components/Dashboard/DashboardTimeline";

import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Tableau de bord | CheckMate",
  description: "Vue d’ensemble des analyses de conformité et accès rapide aux rapports.",
};

export default function DashboardPage() {
  return (
    <>
      <Breadcrumb
        pageName="Tableau de bord"
        description="Consultez l’historique des analyses, les statistiques globales et accédez rapidement à vos rapports."
      />

      <div className="max-w-7xl mx-auto mt-10 space-y-8">
        {/* Statistiques globales */}
        <DashboardStats />

        {/* Tableau des documents analysés */}
        <DashboardTable />

        {/* Timeline des actions */}
        <DashboardTimeline />
      </div>
    </>
  );
}
