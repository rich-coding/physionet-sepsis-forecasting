import { Injectable, signal } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class TurnoSignal {
  // Signal para almacenar el valor del objeto
  private turnoSignal = signal<any | null>(null);

  // Método para obtener el valor actual del signal
  get value(): any | null {
    return this.turnoSignal();
  }

  // Método para actualizar el valor del signal
  set value(newValue: any | null) {
    this.turnoSignal.set(newValue);
  }
}