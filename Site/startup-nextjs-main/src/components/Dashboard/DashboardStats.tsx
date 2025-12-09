"use client";

export default function DashboardStats() {
  const stats = [
    { label: "Documents analysés", value: 12 },
    { label: "Violations détectées", value: 45 },
    { label: "Rapports générés", value: 10 },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {stats.map((s) => (
        <div
          key={s.label}
          className="bg-[var(--color-page-bg-alt)] border border-[var(--color-stroke-stroke)] p-6 rounded-xl shadow-sm text-center"
        >
          <p className="text-3xl font-bold text-[var(--color-primary)]">{s.value}</p>
          <p className="text-sm text-black">{s.label}</p>
        </div>
      ))}
    </div>
  );
}
