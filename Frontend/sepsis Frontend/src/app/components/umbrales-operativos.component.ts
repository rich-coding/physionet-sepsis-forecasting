import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';

@Component({
  selector: 'app-umbrales-operativos',
  standalone: true,
  imports: [CommonModule, MatCardModule],
  templateUrl: './umbrales-operativos.component.html',
  styleUrls: ['./umbrales-operativos.component.scss']
})
export class UmbralesOperativosComponent {}
