/**
 * Autor: Manuela Hernandez
 * Fecha: 20 de septiembre de 2025
 */

export interface HTTPValidationError {
  detail?: ValidationError[];
}

export interface ValidationError {
  loc: (string | number)[];
  msg: string;
  type: string;
}

export interface SepsisBatchRequest {
  batch: SepsisRequest[];
}

export interface SepsisRequest {
  patient_id: string;
  records: SepsisRecord[];
}

export interface SepsisRecord {
  ICULOS: number;
  Temp: number;
  BaseExcess: number;
  DBP: number;
  FiO2: number;
  Gender: number;
  Age: number;
  HCO3: number;
  HR: number;
  HospAdmTime: number;
  Magnesium: number;
  O2Sat: number;
  Resp: number;
}

export interface SepsisScore {
  patient_id: string;
  score: number;
  pred: number;
  threshold: number;
}