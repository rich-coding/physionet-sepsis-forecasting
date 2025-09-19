variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region"
}

variable "image_uri" {
  type        = string
  default     = "728377120672.dkr.ecr.us-east-1.amazonaws.com/sepsis2019-api:latest"
  description = "ECR URI Sepsis"
}

variable "instance_type" {
  type        = string
  default     = "t3.micro"
  description = "EC2 instance type"
}

variable "host_port" {
  type        = number
  default     = 80
  description = "Puerto expuesto en la EC2"
}

variable "app_port" {
  type        = number
  default     = 8080
  description = "Puerto expuesto por el contenedor"
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