#!/usr/bin/env python3
# test_image_publisher.py - Publicador de imágenes de prueba para simulación
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import glob
import time

class TestImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')
        
        # Declarar parámetros
        self.declare_parameter('publish_rate', 1.0)  # Hz (imágenes por segundo)
        self.declare_parameter('image_dir', '')      # Directorio con imágenes
        self.declare_parameter('loop', True)         # Repetir en bucle
        self.declare_parameter('synthetic', True)    # Generar imágenes sintéticas si no hay reales
        
        # Obtener parámetros
        self.publish_rate = self.get_parameter('publish_rate').value
        self.image_dir = self.get_parameter('image_dir').value
        self.loop = self.get_parameter('loop').value
        self.synthetic = self.get_parameter('synthetic').value
        
        # Bridge para convertir imágenes
        self.bridge = CvBridge()
        
        # Publicador de imágenes
        self.publisher = self.create_publisher(
            Image,
            'image_raw',
            10)
            
        # Cargar lista de imágenes
        self.images = []
        self.current_index = 0
        
        if self.image_dir and os.path.isdir(self.image_dir):
            # Buscar imágenes en el directorio
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.images.extend(glob.glob(os.path.join(self.image_dir, ext)))
            
            if self.images:
                self.get_logger().info(f'Cargadas {len(self.images)} imágenes de {self.image_dir}')
            else:
                self.get_logger().warning(f'No se encontraron imágenes en {self.image_dir}')
        else:
            self.get_logger().warning(f'Directorio no válido: {self.image_dir}')
        
        # Si no hay imágenes, generar sintéticas
        if not self.images and self.synthetic:
            self.get_logger().info('Utilizando imágenes sintéticas de prueba')
        
        # Crear temporizador para publicación periódica
        period = 1.0 / self.publish_rate if self.publish_rate > 0 else 1.0
        self.timer = self.create_timer(period, self.publish_image)
        
        self.get_logger().info(f'Publicador de imágenes iniciado (rate: {self.publish_rate} Hz)')
        
    def generate_synthetic_image(self, index):
        """Genera una imagen sintética con huevos para pruebas"""
        # Dimensiones y formato
        width, height = 640, 480
        
        # Crear fondo gris/blanco con variaciones
        base_color = 230 - (index % 3) * 20  # Variar un poco el fondo
        image = np.ones((height, width, 3), dtype=np.uint8) * base_color
        
        # Añadir una sombra parcial
        shadow_width = int(width / 2)
        shadow = np.ones((height, shadow_width, 3), dtype=np.uint8) * (base_color - 20)
        image[:, :shadow_width] = shadow
        
        # Determinar número y posición de huevos basado en el índice
        num_eggs = 1 + (index % 4)  # Entre 1 y 4 huevos
        
        for i in range(num_eggs):
            # Posición aleatoria pero dependiente del índice para reproducibilidad
            x = (100 + (index * 50 + i * 123) % (width - 200))
            y = (100 + (index * 70 + i * 167) % (height - 200))
            
            # Tamaño variable
            major_axis = 40 + (index + i) % 60
            minor_axis = 30 + (index + i) % 40
            
            # Ángulo variable
            angle = (index * 30 + i * 45) % 180
            
            # Color variable pero en el rango marrón
            brown_b = 40 + (index + i) % 30
            brown_g = 50 + (index + i) % 40
            brown_r = 80 + (index + i) % 60
            color = (brown_b, brown_g, brown_r)  # BGR
            
            # Dibujar huevo (elipse)
            cv2.ellipse(image, (x, y), (major_axis, minor_axis), angle, 0, 360, color, -1)
            
            # Añadir un poco de textura/gradiente
            cv2.ellipse(image, (x-5, y-5), (major_axis-10, minor_axis-10), angle, 0, 360, 
                        (brown_b+10, brown_g+10, brown_r+10), -1)
        
        # Añadir un poco de ruido gaussiano
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Desenfocar ligeramente
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        return image
        
    def publish_image(self):
        """Publica la siguiente imagen de la lista o genera una sintética"""
        try:
            # Si hay imágenes reales, usar una de ellas
            if self.images:
                # Verificar índice
                if self.current_index >= len(self.images):
                    if self.loop:
                        self.current_index = 0
                    else:
                        self.get_logger().info('Fin de la secuencia de imágenes')
                        return
                
                # Cargar imagen
                image_path = self.images[self.current_index]
                self.get_logger().debug(f'Publicando imagen: {image_path}')
                
                # Leer imagen
                image = cv2.imread(image_path)
                if image is None:
                    self.get_logger().warning(f'No se pudo leer la imagen: {image_path}')
                    self.current_index += 1
                    return
            else:
                # Generar imagen sintética
                image = self.generate_synthetic_image(self.current_index)
                self.get_logger().debug(f'Publicando imagen sintética #{self.current_index}')
            
            # Convertir a mensaje ROS
            msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            
            # Añadir timestamp y frame_id
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            
            # Publicar
            self.publisher.publish(msg)
            
            # Incrementar índice para la próxima vez
            self.current_index += 1
            
        except Exception as e:
            self.get_logger().error(f'Error publicando imagen: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = TestImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()