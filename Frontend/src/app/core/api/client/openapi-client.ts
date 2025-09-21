/**
 * Autor: Manuela Hernandez
 * Fecha: 20 de septiembre de 2025
 */

import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { SepsisBatchRequest, SepsisScore, PatientsResponse } from '../models/sepsis.models';

@Injectable({
  providedIn: 'root',
})
export class OpenApiClient {

  constructor(private http: HttpClient) {}

  /**
   * Calls the score endpoint to get sepsis scores.
   * @param request The batch request containing patient data.
   * @returns An observable of an array of sepsis scores.
   */
  scoreApiV1ScorePost(request: PatientsResponse): Observable<SepsisScore[]> {
    const url = `/api/v1/score`;
    return this.http.post<SepsisScore[]>(url, request);
  }

  /**
   * Método para obtener los pacientes
   * @returns Un observable con los pacientes por turn
   */
   getPatientsData(number: 1|2|3|4|5): Observable<PatientsResponse> {
    const params = new HttpParams().set('number', String(number));
    // No intentes setear 'Accept-Encoding': el navegador ya lo envía.
    return this.http.get<PatientsResponse>(`/api/v1/turn`, { params });
  }
}
