/**
 * Autor: Manuela Hernandez
 * Fecha: 20 de septiembre de 2025
 */
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { HttpClient } from '@angular/common/http';

/**
 * Servicio para manejar las operaciones relacionadas con el estado de salud del sistema.
 */
@Injectable({
  providedIn: 'root',
})
export class HealthService {
  private readonly baseUrl = 'http://ec2-44-192-130-121.compute-1.amazonaws.com';

  constructor(private http: HttpClient) {}

  /**
   * Método para obtener el estado de salud del sistema.
   * @returns Un observable con el estado de salud.
   */
  getHealth(): Observable<any> {
    const url = `${this.baseUrl}/api/v1/health`;
    return this.http.get<any>(url);
  }
}