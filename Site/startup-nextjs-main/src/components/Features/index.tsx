import SectionTitle from "../Common/SectionTitle";
import SingleFeature from "./SingleFeature";
import featuresData from "./featuresData";

const Features = () => {
  return (
    <section id="fonctionnalites" className="py-16 md:py-20 lg:py-28 bg-[var(--color-page-bg)]">
      <div className="container">
        <SectionTitle
          title="Fonctionnalités principales"
          paragraph="Découvrez comment CheckMate garantit la conformité de vos documents marketing grâce à une analyse intelligente, transparente et basée sur les règles officielles."
          center
        />

        <div className="grid grid-cols-1 gap-x-8 gap-y-14 md:grid-cols-2 lg:grid-cols-3">
          {featuresData.map((feature) => (
            <SingleFeature key={feature.id} feature={feature} />
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
