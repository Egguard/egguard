document.addEventListener('DOMContentLoaded', event => {
    console.log("entro en la pagina")

    // Botones de control de web 
    document.getElementById("btn_con").addEventListener("click", connect) 
    // llama a la función connect() cuando se hace clic en el botón de conexión.
    document.getElementById("btn_dis").addEventListener("click", disconnect)
    // llama a la función disconnect() cuando se hace clic en el botón de desconexión.

    document.getElementById("btn_move").addEventListener("click", move)
    // llama a la función move() cuando se hace clic en el botón de movimiento.
    document.getElementById("btn_stop").addEventListener("click", stop)
    // llama a la función stop() cuando se hace clic en el botón de parada.
    document.getElementById("btn_move_backwards").addEventListener("click", move_backwards)
    // llama a la función move_backwards() cuando se hace clic en el botón de retroceder.
    document.getElementById("btn_left").addEventListener("click", move_left)
    // llama a la función move_left() cuando se hace clic en el botón de girar izquierda.
    document.getElementById("btn_right").addEventListener("click", move_right)
    // llama a la función move_right() cuando se hace clic en el botón de girar derecha.

    data = {
        // ros connection
        ros: null, // manejador de la conexión con el rosbridge_server.
        rosbridge_address: 'ws://127.0.0.1:9090/', // dirección del rosbridge_server.
        connected: false, // estado de la conexión
        
        // service information 
        service_busy: false, 
        service_response: ''
    }

    function call_delante_service(valor){
        data.service_busy = true
        data.service_response = ''	
    
      //definimos los datos del servicio
        let service = new ROSLIB.Service({
            ros: data.ros,
            name: '/movement',
            serviceType: 'custom_interface/srv/MyMoveMsg'
        })
    
        let request = new ROSLIB.ServiceRequest({
            move: valor
        })
    
        service.callService(request, (result) => {
            data.service_busy = false
            data.service_response = JSON.stringify(result)
        }, (error) => {
            data.service_busy = false
            console.error(error)
        })	
    }

    function connect(){ // conectar
	      console.log("Clic en connect")

        // Obtener la dirección del rosbridge_server desde la entrada
        data.rosbridge_address = document.getElementById("rosbridge_address").value

	      data.ros = new ROSLIB.Ros({
                url: data.rosbridge_address
        })

        // Define callbacks
        data.ros.on("connection", () => {
            data.connected = true
            console.log("Conexion con ROSBridge correcta")
            document.getElementById("connection_status").innerText = "Estado: Conectado"
            subscribe_position()
        })
        data.ros.on("error", (error) => {
            console.log("Se ha producido algun error mientras se intentaba realizar la conexion")
            console.log(error)
        })
        data.ros.on("close", () => {
            data.connected = false
            console.log("Conexion con ROSBridge cerrada")
        })
    }

    function disconnect(){ // desconectar
	      data.ros.close()
	      data.connected = false
        console.log('Clic en botón de desconexión')
        document.getElementById("connection_status").innerText = "Estado: Desconectado"
    }

    function move() { // hacia delante
        let topic = new ROSLIB.Topic({
            ros: data.ros,
            name: '/cmd_vel',
            messageType: 'geometry_msgs/msg/Twist'
        })
        let message = new ROSLIB.Message({
            linear: {x: 0.5, y: 0, z: 0, },
            angular: {x: 0, y: 0, z: 0, },
        })
        topic.publish(message)
    }

    function stop() { // parar
        let topic = new ROSLIB.Topic({
            ros: data.ros,
            name: '/cmd_vel',
            messageType: 'geometry_msgs/msg/Twist'
        })
        let message = new ROSLIB.Message({
            linear: {x: 0.0, y: 0, z: 0, },
            angular: {x: 0, y: 0, z: 0.0, },
        })
        topic.publish(message)
    }

    function move_backwards() { // hacia atras
        let topic = new ROSLIB.Topic({
            ros: data.ros,
            name: '/cmd_vel',
            messageType: 'geometry_msgs/msg/Twist'
        })
        let message = new ROSLIB.Message({
            linear: {x: -0.2, y: 0, z: 0, },
            angular: {x: 0, y: 0, z: 0.0, },
        })
        topic.publish(message)
    }

    function move_left() { // girar izquierda
        let topic = new ROSLIB.Topic({
            ros: data.ros,
            name: '/cmd_vel',
            messageType: 'geometry_msgs/msg/Twist'
        })
        let message = new ROSLIB.Message({
            linear: {x: 0, y: 0, z: 0, },
            angular: {x: 0, y: 0, z: 0.3, },
        })
        topic.publish(message)
    }

    function move_right() { // girar derecha
        let topic = new ROSLIB.Topic({
            ros: data.ros,
            name: '/cmd_vel',
            messageType: 'geometry_msgs/msg/Twist'
        })
        let message = new ROSLIB.Message({
            linear: {x: 0, y: 0, z: 0, },
            angular: {x: 0, y: 0, z: -0.3, },
        })
        topic.publish(message)
    }

    function subscribe_position(){ // suscribirse a la posición (se llama cuando se conecta)

        let topic = new ROSLIB.Topic({
            ros: data.ros,
            name: '/odom',
            messageType: 'nav_msgs/msg/Odometry'
        })
    
        topic.subscribe((message) => {
            data.position = message.pose.pose.position
                document.getElementById("pos_x").innerHTML = data.position.x.toFixed(2)
                document.getElementById("pos_y").innerHTML = data.position.y.toFixed(2)
        })

    }

    // Versión usando librería MJPEGCANVAS (requiere cargarla)
    function setCamera(){
        console.log("setting the camera")
      var viewer = new MJPEGCANVAS.Viewer({
          divID : 'mjpeg',
          host : 'localhost',
          width : 640,
          height : 480,
          topic : '/camera/image_raw',
          interval : 200
        })
    }
    
    // otro ejemplo de función (simple para prueba inicial)
    function updateCameraFeed() {
      const img = document.getElementById("cameraFeed");
      const timestamp = new Date().getTime(); // Evita caché agregando un timestamp
      img.src = `http://0.0.0.0:8080/stream?topic=/camera/image_raw`;
      //img.src = `http://localhost:8080/stream?topic=/turtlebot3/camera/image_raw&console.log("Cactualizando: http://0.0.0.0:8080/stream?topic=/camera/image_raw)"`
    }
    
});
