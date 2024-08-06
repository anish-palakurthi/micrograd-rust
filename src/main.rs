use std::fmt::*;
use std::fmt;
use std::ops;

struct Value {
    data: f32, 
    grad: f32, 
    prev: Vec<*const Value>,
    operator: String,
    backward: Box<dyn FnMut()>,
}

impl Value {
    fn new(data: f32, prev: Vec<*const Value>, operator: String) -> Self {
        Self {
            data,
            grad: 0.0,
            prev,
            operator,
            backward: Box::new(|| {}),
        }
    }

    fn pow(mut self, rhs: f32) -> Value {
        let pow_data = self.data.powf(rhs);

        let mut res = Value::new(pow_data, vec![&self as *const Value], "**".to_string());

        let backward = {
            move || {
                self.grad += (rhs * self.data.powf(rhs - 1.0)) * res.grad;
            }
        };
        res.backward = Box::new(backward);
        res

    }

    fn relu(mut self) -> Value {
        let mut res = match (self.data < 0.0){
            true => Value::new(0.0, vec![&self as *const Value], "ReLU".to_string()),
            false => Value::new(self.data, vec![&self as *const Value], "ReLU".to_string()),
        };

        let backward = {
            move || {
                self.grad += (res.data > 0.0) as i32 as f32 * res.grad;
            }
        };
        res.backward = Box::new(backward);
        res
    }

    fn build_top(v: &Value, topo: &mut Vec<*const Value>, visited: &mut Vec<*const Value>) {
        if visited.contains(&(v as *const Value)) {
            return;
        }
        visited.push(v as *const Value);
        for prev in v.prev.iter() {
            Self::build_top(unsafe { &**prev }, topo, visited);
        }
        topo.push(v as *const Value);
    }

    fn backward(mut self) {
        let mut topo: Vec<*const Value> = vec![];
        let mut visited: Vec<*const Value> = vec![];

        Self::build_top(&self, &mut topo, &mut visited);
        self.grad = 1.0;

        let reversed_topo = topo.iter().rev();
        for v in reversed_topo {
            (unsafe { &mut **(v as *mut Value) }.backward)();
        }
    }
}

impl ops::Add for Value {
    type Output = Value;

    fn add(mut self, mut rhs: Value) -> Value {
        let sum_data = self.data + rhs.data;

        let mut res = Value::new(sum_data, vec![&self as *const Value, &rhs as *const Value], "+".to_string());

        let backward = {
            let res_grad = res.grad;
            move || {
                self.grad += res_grad;
                rhs.grad += res_grad;
            }
        };

        res.backward = Box::new(backward);
        res
    }
}

impl ops::Mul for Value {
    type Output = Value;

    fn mul(mut self, mut rhs: Value) -> Value {
        let prod_data = self.data * rhs.data;

        let mut res = Value::new(prod_data, vec![&self as *const Value, &rhs as *const Value], "*".to_string());

        let backward = {
            let res_grad = res.grad;
            move || {
                self.grad += rhs.data * res_grad;
                rhs.grad += self.data * res_grad;
            }
        };

        res.backward = Box::new(backward);
        res
    }
}

impl ops::Neg for Value {
    type Output = Value;

    fn neg(mut self) -> Self::Output {
        self.data = -1.0 * self.data;
        self
    }
}

impl ops::Sub<Value> for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        self + (-rhs)
    }
}

impl ops::Div<Value> for Value {
    type Output = Value;

    fn div(self, rhs: Value) -> Self::Output {
        self * rhs.pow(-1.0)
    }
}




impl Display for Value{

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        write!(f, "Data:{}; grad:{};  prev:{:#?};  operator: {}", self.data, self.grad, self.prev, self.operator)
    }
}



fn main(){

    let example = Value::new(1.0, vec![], "+".to_string());

    println!("Example!: {}", example);

}