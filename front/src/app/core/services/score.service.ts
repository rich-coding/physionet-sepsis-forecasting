/**
 * Autor: Manuela Hernandez
 * Fecha: 20 de septiembre de 2025
 */
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { OpenApiClient } from '../api/client/openapi-client';
import { SepsisBatchRequest, SepsisScore } from '../api/models/sepsis.models';
import { nivelRiesgo } from '../../shared/utils/risk.util';

/**
 * Servicio para manejar las operaciones relacionadas con el cálculo de scores de sepsis.
 */
@Injectable({
  providedIn: 'root',
})
export class ScoreService {
  constructor(private apiClient: OpenApiClient) {}

  /**
   * Método para obtener el score de sepsis a partir de un cuerpo de solicitud.
   * @param body Cuerpo de la solicitud que contiene los datos del paciente.
   * @returns Un observable con la respuesta mapeada a un objeto de vista.
   */
  score(body: SepsisBatchRequest): Observable<{ probabilidad: number; riesgo: 'ALTO' | 'MEDIO' | 'BAJO'; mensajes?: string[] }> {
    return this.apiClient.scoreApiV1ScorePost(body).pipe(
      map((response: SepsisScore[]) => {
        const score = response[0]; // Asumimos que solo hay un paciente en el batch
        const probabilidad = score.score;
        let riesgo: 'ALTO' | 'MEDIO' | 'BAJO';
        if (probabilidad >= 0.8) {
          riesgo = 'ALTO';
        } else if (probabilidad >= 0.5) {
          riesgo = 'MEDIO';
        } else {
          riesgo = 'BAJO';
        }
        return { probabilidad, riesgo };
      })
    );
  }

  /**
   * Método para obtener el score de sepsis sin procesar, útil para inspeccionar cabeceras o estado.
   * @param body Cuerpo de la solicitud que contiene los datos del paciente.
   * @returns Un observable con la respuesta cruda del servidor.
   */
  scoreRaw(body: SepsisBatchRequest): Observable<SepsisScore[]> {
    return this.apiClient.scoreApiV1ScorePost(body);
  }
}

export const UMBRAL_HIGH = 0.8;
export const UMBRAL_MEDIUM = 0.7;
export const REGLA_OPERACION = { raise: 0.85, hold: 0.8, clear: 0.6, clearHours: 3 } as const;

/**
 * Helper para transformar la respuesta del endpoint a la vista del tablero.
 * @param response Respuesta del endpoint /api/v1/score.
 * @returns Objeto transformado con probabilidad y nivel de riesgo.
 */
export function transformarRespuesta(response: SepsisScore[]): { probabilidad: number; riesgo: 'ALTO' | 'MEDIO' | 'BAJO' } {
  const score = response[0]; // Asumimos que solo hay un paciente en el batch
  const probabilidad = score.score;
  const riesgo = nivelRiesgo(probabilidad, UMBRAL_HIGH, UMBRAL_MEDIUM);
  return { probabilidad, riesgo };
}