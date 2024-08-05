use std::collections::HashSet;
use std::fmt::*;
use std::fmt;
struct Value {
    data: f32, 
    grad: f32, 
    backward: f32,
    prev: HashSet<usize>,
    operator: String 
}

impl Value {
    fn new(data: f32, grad: f32, backward: f32, prev: Vec<usize>, operator: String) -> Self{
        Self {
            data,
            grad,
            backward,
            prev: prev.into_iter().collect(),
            operator

        }
    }

}
impl Display for Value{

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        write!(f, "Data:{}; grad:{};  backward:{};  prev:{:#?};  operator: {}", self.data, self.grad, self.backward, self.prev, self.operator)

    }

}



fn main(){

    let example = Value::new(1.0, 2.0, 3.0, vec![1,2], "+".to_string());

    println!("Example!: {}", example);



}