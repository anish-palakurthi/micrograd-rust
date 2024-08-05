use std::collections::HashSet;
use std::fmt::*;
use std::fmt;
use std::ops;

struct Value {
    data: f32, 
    grad: f32, 
    prev: Vec<usize>,
    operator: String,
    backward: Box<dyn FnMut()>,
}

impl Value {
    fn new(data: f32, prev: Vec<usize>, operator: String) -> Self {
        Self {
            data,
            grad: 0.0,
            prev,
            operator,
            backward: Box::new(|| {}),
        }
    }
}

impl ops::Add<Value> for Value {
    type Output = Value;

    fn add(mut self, mut rhs: Value) -> Value {
        let sum_data = self.data + rhs.data;

        let mut res = Value::new(sum_data, vec![&self as *const _ as usize, &rhs as *const _ as usize],  "+".to_string());

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

impl Display for Value{

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        write!(f, "Data:{}; grad:{};  prev:{:#?};  operator: {}", self.data, self.grad, self.prev, self.operator)
    }
}



fn main(){

    let example = Value::new(1.0, vec![1, 2], "+".to_string());

    println!("Example!: {}", example);

}