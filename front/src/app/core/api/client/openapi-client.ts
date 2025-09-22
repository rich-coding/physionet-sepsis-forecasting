/**
 * Autor: Manuela Hernandez
 * Fecha: 20 de septiembre de 2025
 */

import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { SepsisBatchRequest, SepsisScore } from '../models/sepsis.models';

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
  scoreApiV1ScorePost(request: any): Observable<SepsisScore[]> {
    const url = `/api/v1/score`;
    return this.http.post<SepsisScore[]>(url, request);
  }

  /**
   * Método para obtener los datos del archivo mock.json.
   * @returns Un observable con los datos del archivo mock.json.
   */
  getPatientsData(number: string): Observable<any> {
    const headers = new HttpHeaders({
      Accept: 'application/json',
      'Accept-Encoding': 'gzip, deflate, br',
      'Content-Type': 'application/json'
    })
    
    const params = new HttpParams().set('number', String(number));
    return this.http.get<any>(`/api/v1/turn`, { headers, params });
  }
}
