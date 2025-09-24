output "public_ip" {
  value = aws_instance.app.public_ip
}

output "public_dns" {
  value = aws_instance.app.public_dns
}

output "url" {
  value = "http://${aws_instance.app.public_dns}:${var.app_port}"
}

