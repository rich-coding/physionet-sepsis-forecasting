import { Injectable } from '@angular/core';
import {
  HttpEvent,
  HttpHandler,
  HttpInterceptor,
  HttpRequest,
  HttpErrorResponse,
} from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';

/**
 * Clase para interceptar errores HTTP y mapearlos a un ApiError tipado.
 */
@Injectable()
export class ErrorInterceptor implements HttpInterceptor {
  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    return next.handle(req).pipe(
      catchError((error: HttpErrorResponse) => {
        let errorMessage = 'Ocurrió un error desconocido';
        if (error.error instanceof ErrorEvent) {
          // Error del cliente
          errorMessage = `Error del cliente: ${error.error.message}`;
        } else {
          // Error del servidor
          errorMessage = `Error del servidor: ${error.status} - ${error.message}`;
        }
        return throwError(() => new ApiError(errorMessage));
      })
    );
  }
}

/**
 * Clase para representar errores de la API con un mensaje claro.
 */
export class ApiError {
  constructor(public message: string) {}
}