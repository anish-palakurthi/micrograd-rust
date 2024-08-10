use std::fmt::*;
use std::fmt;
use std::ops;
use std::collections::HashSet;
pub struct Value {
    data: f32, 
    grad: f32, 
    prev: Vec<*const Value>,
    operator: String,
    backward: Box<dyn FnMut()>,
}

impl Value {
    pub fn new(data: f32, prev: Vec<*const Value>, operator: String) -> Self {
        Self {
            data,
            grad: 0.0,
            prev,
            operator,
            backward: Box::new(|| {}),
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
        res.backward = Box::new(backward);
        res

    }

    pub fn relu(mut self) -> Value {
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



pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    nonlin: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool) -> Self {
        let mut rng = rand::thread_rng();
        let w = (0..nin).map(|_| Value::new(rng.gen_range(-1.0..1.0), vec![], "".to_string())).collect();
        let b = Value::new(0.0, vec![], "".to_string());
        Self { w, b, nonlin }
    }

    pub fn call(&self, x: &[Value]) -> Value {
        let act = self.w.iter().zip(x).fold(self.b.clone(), |acc, (wi, xi)| acc + (wi.clone() * xi.clone()));
        if self.nonlin {
            act.relu()
        } else {
            act
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<&Value> {
        let mut params = self.w.iter().collect::<Vec<&Value>>();
        params.push(&self.b);
        params
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}Neuron({})", if self.nonlin { "ReLU" } else { "Linear" }, self.w.len())
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin: bool) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin, nonlin)).collect();
        Self { neurons }
    }

    pub fn call(&self, x: &[Value]) -> Vec<Value> {
        let out: Vec<Value> = self.neurons.iter().map(|n| n.call(x)).collect();
        if out.len() == 1 {
            vec![out[0].clone()]
        } else {
            out
        }
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<&Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Layer of [{}]", self.neurons.iter().map(|n| n.to_string()).collect::<Vec<String>>().join(", "))
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: Vec<usize>) -> Self {
        let mut sz = vec![nin];
        sz.extend(nouts.iter());
        let layers = (0..nouts.len()).map(|i| Layer::new(sz[i], sz[i + 1], i != nouts.len() - 1)).collect();
        Self { layers }
    }

    pub fn call(&self, x: &[Value]) -> Vec<Value> {
        let mut x = x.to_vec();
        for layer in &self.layers {
            x = layer.call(&x);
        }
        x
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<&Value> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }
}

impl fmt::Display for MLP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MLP of [{}]", self.layers.iter().map(|layer| layer.to_string()).collect::<Vec<String>>().join(", "))
    }
}


