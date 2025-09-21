import { Component, inject, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';
import { UmbralesOperativosComponent } from './umbrales-operativos.component';
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';
import { OpenApiClient } from '../core/api/client/openapi-client';
import { MatSelectChange, MatSelectModule } from '@angular/material/select';

const CHIP_MAP: any = {
  ICULOS: 'Tiempo UCI',
  Temp: 'Temperatura',
  BaseExcess: 'Exceso Bicarbonato',
  DBP: 'Presión Diastólica',
  FiO2: 'Oxygeno',
  Gender: 'Genero',
  Age: 'Edad',
  HCO3: 'Bicarbonatos',
  HR: 'Ritmo Cardíaco',
  HospAdmTime: 'Hor. Ing. UCI',
  Magnesium: 'Magnesio',
  O2Sat: 'Saturación O2',
  Resp: 'Respiración',
};

@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.scss'],
  imports: [
    CommonModule,
    MatSidenavModule,
    MatListModule,
    MatIconModule,
    MatCardModule,
    UmbralesOperativosComponent,
    MatTableModule,
    MatButtonModule,
    MatSelectModule
  ],
})
export class SidebarComponent implements OnInit {
  readonly openapiClient = inject(OpenApiClient);
  pacienteSeleccionado: any;

  displayedColumns: string[] = [
    'id',
    'riesgo',
    'tendencia',
    'hora',
    'signos',
    'estado',
    'acciones',
  ];
  pacientes: any[] = [];

  constructor() {}
  ngOnInit(): void {
  }

  obtenerDatos() {
    if(this.pacienteSeleccionado) {
      this.openapiClient.scoreApiV1ScorePost(this.pacienteSeleccionado.datos).subscribe((data) => {
        console.log(data);
      });
    }
    
  }

  seleccionarTurnos(turno: MatSelectChange) {
    const numeroTurno = turno.value;

    this.openapiClient.getPatientsData(numeroTurno).subscribe((data: any) => {

      this.pacientes = data.batch.map((paciente: any) => ({
        id: paciente.patient_id,
        //riesgo: paciente.score,
        //tendencia: paciente.score >= 85 ? 'ALTO' : paciente.score >= 80 ? 'MEDIO' : 'BAJO',
        //hora: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        signos: this.obtenerSignos(paciente.records),
        estado: Math.random() > 0.5 ? 'NUEVA' : 'VIGILANCIA',
        datos: paciente.records
      }));
    });
    
  }

  obtenerSignos(datosPaciente: any) {
    return Object.keys(datosPaciente).reduce((acc: any[], key) => {
      if (CHIP_MAP[key]) {
        acc.push({
          label: `${CHIP_MAP[key]} ${datosPaciente[key]}`,
          tipo: CHIP_MAP[key] || 'info',
          value: datosPaciente[key],
        });
      }

      return acc;
    }, []);
  }

  seleccionarPaciente(paciente: any) {
    this.pacienteSeleccionado = paciente;
    this.obtenerDatos();
  }
}
