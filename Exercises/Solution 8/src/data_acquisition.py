from tdmclient import ClientAsync

thymio_data = []
acquire_data = False


def motors(left, right):
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }

def record_data():
    thymio_data.append({"ground":th["prox.ground.reflected"],
                        "sensor":th["prox.ground.reflected"],
                        "left_speed":th["motor.left.speed"],
                        "right_speed":th["motor.right.speed"]})

def on_variables_changed(node, variables):
    try:
        prox = variables["prox.horizontal"]
        prox_front = prox[2]
        speed = -prox_front // 10
        node.send_set_variables(motors(speed, speed))
    except KeyError:
        pass  # prox.horizontal not found

async def get_data():
with ClientAsync() as client:
    async def prog():
        with await client.lock() as node:
            await node.watch(variables=True)
            node.add_variables_changed_listener(on_variables_changed)
            await client.sleep()
    client.run_async_program(prog)
