export type NivelRiesgo = 'ALTO' | 'MEDIO' | 'BAJO';

export function nivelRiesgo(p: number, high = 0.80, medium = 0.70): NivelRiesgo {
  if (p >= high) return 'ALTO';
  if (p >= medium) return 'MEDIO';
  return 'BAJO';
}

export interface PoliticaAlertas {
  raise: number;   // encender
  hold: number;    // mantener
  clear: number;   // resolver
  clearHours: number; // horas consecutivas por debajo de 'clear' para cerrar
}