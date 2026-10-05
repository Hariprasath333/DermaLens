import type { LesionClassCode } from "../types/lesioniq";

export const lesionClasses: Record<LesionClassCode, string> = {
  MEL: "Melanoma",
  NV: "Melanocytic nevus",
  BCC: "Basal cell carcinoma",
  BKL: "Benign keratosis",
  AK: "Actinic keratosis",
  SCC: "Squamous cell carcinoma",
  VASC: "Vascular lesion",
  DF: "Dermatofibroma"
};

export const lesionClassOrder: LesionClassCode[] = ["MEL", "NV", "BCC", "BKL", "AK", "SCC", "VASC", "DF"];

