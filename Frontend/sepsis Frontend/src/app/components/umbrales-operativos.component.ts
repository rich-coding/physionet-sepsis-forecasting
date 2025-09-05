import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';

@Component({
  selector: 'app-umbrales-operativos',
  standalone: true,
  imports: [CommonModule, MatCardModule, MatTableModule, MatButtonModule],
  templateUrl: './umbrales-operativos.component.html',
  styleUrls: ['./umbrales-operativos.component.scss']
})
export class UmbralesOperativosComponent {
  displayedColumns: string[] = [
    'id', 'riesgo', 'tendencia', 'hora', 'signos', 'estado', 'acciones'
  ];
  pacientes = [
    {
      id: '00432',
      riesgo: '0.87',
      tendencia: 'ALTO',
      hora: '10:05',
      signos: [
        { label: 'HR 110', tipo: 'hr' },
        { label: 'MAP 58', tipo: 'map' },
        { label: 'Temp 38.5', tipo: 'temp' },
        { label: 'SpO2 91%', tipo: 'spo2' }
      ],
      estado: 'NUEVA'
    },
    {
      id: '00318',
      riesgo: '0.82',
      tendencia: 'ALTO',
      hora: '09:55',
      signos: [
        { label: 'HR 104', tipo: 'hr' },
        { label: 'MAP 63', tipo: 'map' },
        { label: 'Resp 26', tipo: 'resp' }
      ],
      estado: 'VIGILANCIA'
    },
    {
      id: '00901',
      riesgo: '0.74',
      tendencia: 'MEDIO',
      hora: '10:10',
      signos: [
        { label: 'HR 95', tipo: 'map' },
        { label: 'MAP 70', tipo: 'map' }
      ],
      estado: 'VIGILANCIA'
    },
    {
      id: '00077',
      riesgo: '0.71',
      tendencia: 'MEDIO',
      hora: '10:12',
      signos: [
        { label: 'Temp 38.2', tipo: 'temp' },
        { label: 'Resp 24', tipo: 'resp' }
      ],
      estado: 'VIGILANCIA'
    },
    {
      id: '00550',
      riesgo: '0.41',
      tendencia: 'BAJO',
      hora: '10:07',
      signos: [
        { label: 'HR 82', tipo: 'map' },
        { label: 'MAP 82', tipo: 'map' }
      ],
      estado: 'NUEVA'
    }
  ];
}
