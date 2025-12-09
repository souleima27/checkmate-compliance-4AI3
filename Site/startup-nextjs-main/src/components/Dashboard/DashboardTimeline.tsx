"use client";

export default function DashboardTimeline() {
  const timeline = [
    { time: "16:20", action: "Parsing terminé" },
    { time: "16:22", action: "Violations détectées" },
    { time: "16:30", action: "Corrections proposées" },
    { time: "16:35", action: "Rapport final généré" },
  ];

  return (
    <div className="bg-[var(--color-page-bg-alt)] border border-[var(--color-stroke-stroke)] rounded-xl shadow-sm p-6">
      <h2 className="text-xl font-semibold mb-4 text-black">Timeline des actions</h2>
      <ul className="space-y-3">
        {timeline.map((t, i) => (
          <li key={i} className="flex items-center gap-4 text-black">
            <span className="font-bold text-[var(--color-primary)]">{t.time}</span>
            <span>{t.action}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
