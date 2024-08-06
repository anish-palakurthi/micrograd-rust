use std::fmt::*;
use std::fmt;
use std::ops;
use std::collections::HashSet;
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

    fn build_topo(v: &Value, topo: &mut Vec<*const Value>, visited: &mut HashSet<*const Value>) {
        if !visited.insert(v as *const Value) {
            return;
        }
        for prev in &v.prev {
            Self::build_topo(unsafe { &**prev }, topo, visited);
        }
        topo.push(v as *const Value);
    }
    
    fn backward(&mut self) {
        let mut topo: Vec<*const Value> = Vec::new();
        let mut visited: HashSet<*const Value> = HashSet::new();
    
        Self::build_topo(self, &mut topo, &mut visited);
        self.grad = 1.0;
    
        for &v in topo.iter().rev() {
            unsafe {
                (&mut *(v as *mut Value)).backward();
            }
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