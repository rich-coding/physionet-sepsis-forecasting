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
import { TurnoSignal } from '../shared/signals/turno.signal';

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
    MatSelectModule,
  ],
})
export class SidebarComponent implements OnInit {
  readonly openapiClient = inject(OpenApiClient);
  pacienteSeleccionado: any;
  datosScore: any;
  numeroTurnoSeleccionado: string = '';

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

  constructor(private turnoSignal: TurnoSignal) {}
  ngOnInit(): void {}

  obtenerDatos(datosPacientes: any) {
    this.openapiClient
      .scoreApiV1ScorePost(datosPacientes)
      .subscribe((data) => {
        this.datosScore = data;
        this.pacientes = datosPacientes.batch.map((paciente: any) => {
          const datosScorePaciente = data.find((d: any) => d.patient_id === paciente.patient_id);
          return {
            id: paciente.patient_id,
            riesgo: datosScorePaciente?.score || 0,
            tendencia: datosScorePaciente && datosScorePaciente.score >= 0.85 ? 'ALTO' : datosScorePaciente && datosScorePaciente.score >= 0.80 ? 'MEDIO' : 'BAJO',
            hora: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            signos: this.obtenerSignos(paciente.records),
            estado: Math.random() > 0.5 ? 'NUEVA' : 'VIGILANCIA',
            datos: paciente.records[0],
            pred: datosScorePaciente?.pred || 0
          };
        });

        this.pacientes.sort((a, b) => b.riesgo - a.riesgo);
        this.turnoSignal.value = this.pacientes;
      });
  }

  filtrarPorRiesgo(riesgo: string) {
    return this.pacientes.filter(paciente => paciente.tendencia === riesgo);
  }

  seleccionarTurnos(turno: MatSelectChange) {
    this.numeroTurnoSeleccionado = turno.value;

    this.openapiClient.getPatientsData(this.numeroTurnoSeleccionado).subscribe((data: any) => {
      this.obtenerDatos(data);
    });
  }

  obtenerSignos(datosPaciente: any) {
    const datPaciente = datosPaciente[0] ;
    return Object.keys(datPaciente).reduce((acc: any[], key) => {
      if (CHIP_MAP[key]) {
        acc.push({
          label: `${CHIP_MAP[key]} ${datPaciente[key]}`,
          tipo: CHIP_MAP[key] || 'info',
          value: datPaciente[key] || '',
        });
      }

      return acc;
    }, []);
  }

  seleccionarPaciente(paciente: any) {
    this.pacienteSeleccionado = paciente;
  }
}
