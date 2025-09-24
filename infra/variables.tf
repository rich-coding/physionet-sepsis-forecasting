variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region"
}

variable "image_uri_api" { 
  type        = string
  default     = "728377120672.dkr.ecr.us-east-1.amazonaws.com/sepsis2019-api:latest"
  description = "ECR URI Sepsis API"
}

variable "image_uri_frontend" { 
  type        = string
  default     = "728377120672.dkr.ecr.us-east-1.amazonaws.com/sepsis2019-frontend:latest"
  description = "ECR URI Sepsis Frontend"
}

variable "instance_type" {
  type        = string
  default     = "t3.micro"
  description = "EC2 instance type"
}

variable "host_port_api" {
  type        = number
  default     = 8081
  description = "Puerto host para la API"
}

variable "host_port_front" {
  type        = number
  default     = 8080
  description = "Puerto host para el Frontend"
}

variable "app_port" {
  type        = number
  default     = 80
  description = "Puerto expuesto por el contenedor (interno)"
}

variable "key_name" {
  type        = string
  default     = "FirstEC2InstancePairKey"
  description = "Nombre del key pair para SSH"
}

variable "instance_profile_name" { 
  type = string
  default = "LabInstanceProfile" 
}

variable "root_volume_size" {
  type        = number
  default     = 20  
  description = "Tamaño del volumen raíz (GiB)"
}

variable "compose_project" {
  type        = string
  default     = "sepsis"
  description = "Nombre del proyecto docker-compose"
}