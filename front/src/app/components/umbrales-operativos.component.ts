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
  
}
