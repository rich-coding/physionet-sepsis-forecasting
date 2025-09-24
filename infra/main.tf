terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

locals {
  registry_api = split("/", var.image_uri_api)[0]  
  registry_frontend = split("/", var.image_uri_frontend)[0]  
}

provider "aws" {
  region = var.aws_region
}

# VPC/subnets por defecto
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# SG: permite entrar por host_port
resource "aws_security_group" "app_sg" {
  name        = "ec2-docker-app-sg"
  description = "Allow inbound on host_port and all outbound"
  vpc_id      = data.aws_vpc.default.id
  
  ingress {
    from_port         = 22
    to_port           = 22
    protocol          = "tcp"
    cidr_blocks       = ["0.0.0.0/0"]
    ipv6_cidr_blocks  = ["::/0"]
    description       = "Frontend"
  }

  # Front
  ingress {
    from_port         = var.host_port_front
    to_port           = var.host_port_front
    protocol          = "tcp"
    cidr_blocks       = ["0.0.0.0/0"]
    ipv6_cidr_blocks  = ["::/0"]
    description       = "Frontend"
  }

  # API
  ingress {
    from_port         = var.host_port_api
    to_port           = var.host_port_api
    protocol          = "tcp"
    cidr_blocks       = ["0.0.0.0/0"]
    ipv6_cidr_blocks  = ["::/0"]
    description       = "API"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = { Name = "ec2-docker-app-sg" }
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_instance_profile" "existing" {
  name = var.instance_profile_name
}


# AMI Amazon Linux 2023
data "aws_ami" "al2023" {
  owners      = ["137112412989"] # Amazon
  most_recent = true

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# EC2
resource "aws_instance" "app" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  subnet_id              = element(data.aws_subnets.default.ids, 0)
  vpc_security_group_ids = [aws_security_group.app_sg.id]
  key_name               = var.key_name != "" ? var.key_name : null
  iam_instance_profile   = data.aws_iam_instance_profile.existing.name
  associate_public_ip_address = true
  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    region          = var.aws_region
    compose_content = file(abspath("${path.root}/../docker-compose-prod.yml"))
    image_uri_api   = var.image_uri_api
    image_uri_front = var.image_uri_frontend
    host_port_api   = var.host_port_api
    host_port_front = var.host_port_front
    container_port  = var.app_port
    compose_project = var.compose_project
    registry_api    = local.registry_api
    registry_front  = local.registry_frontend
  })
  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
  }
  tags = { Name = "ec2-docker-app" }
}
