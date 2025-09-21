import { Component } from '@angular/core';
import { SidebarComponent } from "./sidebar.component";
import { HeaderComponent } from "./header.component";
import { UmbralesOperativosComponent } from './umbrales-operativos.component';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
  imports: [SidebarComponent, HeaderComponent, UmbralesOperativosComponent]
})
export class DashboardComponent {}
