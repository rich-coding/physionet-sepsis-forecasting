FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci


COPY . .
RUN npm run build -- --configuration production --base-href / --deploy-url /
RUN echo ">>> dist tree" && ls -la /app/dist && ls -la /app/dist/sepsis || true

FROM nginx:stable-alpine
RUN rm -f /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist/sepsis/browser /usr/share/nginx/html

EXPOSE 80
