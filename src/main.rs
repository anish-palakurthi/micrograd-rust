use std::fmt::*;
use std::fmt;
use std::ops;
use std::sync::{Mutex, Arc};
use std::collections::HashSet;

#[derive(Clone)]
pub struct Value {
    data: f32, 
    grad: f32, 
    prev: Vec<*const Value>,
    operator: String,
    backward: Arc<Mutex<dyn FnMut() + Send>>,
}

impl Value {
    pub fn new(data: f32, prev: Vec<*const Value>, operator: String) -> Self {
        Self {
            data,
            grad: 0.0,
            prev,
            operator,
            backward: Arc::new(Mutex::new(|| {})),
        }
    }

    pub fn pow(mut self, rhs: f32) -> Value {
        let pow_data = self.data.powf(rhs);

        let mut res = Value::new(pow_data, vec![&self as *const Value], "**".to_string());

        let backward = {
            move || {
                self.grad += (rhs * self.data.powf(rhs - 1.0)) * res.grad;
            }
        };
        res.backward = Arc::new(Mutex::new(backward));
        res
    }

    pub fn relu(mut self) -> Value {
        let mut res = match self.data < 0.0 {
            true => Value::new(0.0, vec![&self as *const Value], "ReLU".to_string()),
            false => Value::new(self.data, vec![&self as *const Value], "ReLU".to_string()),
        };

        let backward = {
            move || {
                self.grad += (res.data > 0.0) as i32 as f32 * res.grad;
            }
        };
        res.backward = Arc::new(Mutex::new(backward));
        res
    }
    pub fn build_topo(v: &Value, topo: &mut Vec<*const Value>, visited: &mut HashSet<*const Value>) {
        if !visited.insert(v as *const Value) {
            return;
        }
        for prev in &v.prev {
            Self::build_topo(unsafe { &**prev }, topo, visited);
        }
        topo.push(v as *const Value);
    }
    
    pub fn backward(&mut self) {
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

        res.backward = Arc::new(Mutex::new(backward));
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

        res.backward = Arc::new(Mutex::new(backward));
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

fn main(){}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_addition() {
        let a = Value::new(2.0, vec![], "".to_string());
        let b = Value::new(3.0, vec![], "".to_string());
        let c = a + b;
        assert_eq!(c.data, 5.0);
    }

    #[test]
    fn test_value_multiplication() {
        let a = Value::new(2.0, vec![], "".to_string());
        let b = Value::new(3.0, vec![], "".to_string());
        let c = a * b;
        assert_eq!(c.data, 6.0);
    }

    #[test]
    fn test_value_negation() {
        let a = Value::new(2.0, vec![], "".to_string());
        let b = -a;
        assert_eq!(b.data, -2.0);
    }

    #[test]
    fn test_value_subtraction() {
        let a = Value::new(5.0, vec![], "".to_string());
        let b = Value::new(3.0, vec![], "".to_string());
        let c = a - b;
        assert_eq!(c.data, 2.0);
    }

    #[test]
    fn test_value_division() {
        let a = Value::new(6.0, vec![], "".to_string());
        let b = Value::new(3.0, vec![], "".to_string());
        let c = a / b;
        assert_eq!(c.data, 2.0);
    }

    #[test]
    fn test_value_relu() {
        let a = Value::new(-1.0, vec![], "".to_string());
        let b = a.relu();
        assert_eq!(b.data, 0.0);

        let c = Value::new(1.0, vec![], "".to_string());
        let d = c.relu();
        assert_eq!(d.data, 1.0);
    }

    #[test]
    fn test_value_pow() {
        let a = Value::new(2.0, vec![], "".to_string());
        let b = a.pow(3.0);
        assert_eq!(b.data, 8.0);
    }

    #[test]
    fn test_more_ops() {
        let mut a = Value::new(-4.0, vec![], "".to_string());
        let mut b = Value::new(2.0, vec![], "".to_string());
        let mut c = a.clone() + b.clone();
        let mut d = a.clone() * b.clone() + b.clone().pow(3.0);
        c = c.clone() + c.clone() + Value::new(1.0, vec![], "".to_string());
        c = c.clone() + Value::new(1.0, vec![], "".to_string()) + c.clone() + (-a.clone());
        d = d.clone() + d.clone() * Value::new(2.0, vec![], "".to_string()) + (b.clone() + a.clone()).relu();
        d = d.clone() + Value::new(3.0, vec![], "".to_string()) * d.clone() + (b.clone() - a.clone()).relu();
        let e = c.clone() - d.clone();
        let f = e.clone().pow(2.0);
        let mut g = f.clone() / Value::new(2.0, vec![], "".to_string());
        g = g.clone() + Value::new(10.0, vec![], "".to_string()) / f.clone();
        g.backward();

        let amg = a;
        let bmg = b;
        let gmg = g;

        
        // forward pass went well
        println!("gmg.data: {}", gmg.data);
        println!("amg.grad: {}", amg.grad);
        println!("bmg.grad: {}", bmg.grad);
    }
}


