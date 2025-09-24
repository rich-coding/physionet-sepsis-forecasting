/**
 * Autor: Manuela Hernandez
 * Fecha: 20 de septiembre de 2025
 */

import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { SepsisBatchRequest, SepsisScore } from '../models/sepsis.models';

@Injectable({
  providedIn: 'root',
})
export class OpenApiClient {
  private readonly baseUrl = 'http://ec2-3-215-142-220.compute-1.amazonaws.com';
  private readonly mockUrl = '/assets/mock_'; // Ruta al archivo mock.json
  constructor(private http: HttpClient) {}

  /**
   * Calls the score endpoint to get sepsis scores.
   * @param request The batch request containing patient data.
   * @returns An observable of an array of sepsis scores.
   */
  scoreApiV1ScorePost(request: any, turno: string): Observable<SepsisScore[]> {
    const url = `${this.mockUrl}score_${turno}.json`;
    return this.http.get<SepsisScore[]>(url);
  }

  /**
   * Método para obtener los datos del archivo mock.json.
   * @returns Un observable con los datos del archivo mock.json.
   */
  getPatientsData(turno: string): Observable<any> {
    //const url = `${this.baseUrl}/api/v1/turn?number=${turno}`;
    return this.http.get<any>(`${this.mockUrl}${turno}.json`, {
      headers: new HttpHeaders({
        Accept: 'application/json',
        'Accept-Encoding': 'gzip, deflate, br',
        'Content-Type': 'application/json'
      }),
      timeout: 60000,
      keepalive: true
    });
  }
}
