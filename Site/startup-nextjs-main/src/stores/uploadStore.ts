import { create } from "zustand";

interface MetadataType {
  "Société de Gestion": string;
  "Est ce que le produit fait partie de la Sicav d'Oddo": boolean;
  "Le client est-il un professionnel": boolean;
  "Le document fait-il référence à une nouvelle Stratégie": boolean;
  "Le document fait-il référence à un nouveau Produit": boolean;
  [key: string]: string | boolean;
}

interface UploadState {
  marketingFile: File | null;
  prospectusFile: File | null;
  metadata: MetadataType;
  setMarketingFile: (file: File | null) => void;
  setProspectusFile: (file: File | null) => void;
  setMetadata: (metadata: MetadataType) => void;
}

const defaultMetadata: MetadataType = {
  "Société de Gestion": "ODDO BHF ASSET MANAGEMENT SAS",
  "Est ce que le produit fait partie de la Sicav d'Oddo": false,
  "Le client est-il un professionnel": false,
  "Le document fait-il référence à une nouvelle Stratégie": false,
  "Le document fait-il référence à un nouveau Produit": true,
};

export const useUploadStore = create<UploadState>((set) => ({
  marketingFile: null,
  prospectusFile: null,
  metadata: defaultMetadata,
  setMarketingFile: (file) => set({ marketingFile: file }),
  setProspectusFile: (file) => set({ prospectusFile: file }),
  setMetadata: (metadata) => set({ metadata }),
}));
