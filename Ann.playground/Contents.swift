import UIKit

var str = "Hello, playground"

postfix operator ^
struct Matrix<T: Numeric> {
	var rows: Int
	var columns: Int
	var data: [T]
	
	init(rows: Int = 0, columns: Int = 0) {
		self.rows = rows
		self.columns = columns
		self.data = Array(repeating: 0, count: rows * columns)
	}
	
	init(data: [T], rows: Int, columns: Int) {
		self.data = data
		self.rows = rows
		self.columns = columns
	}
	
	mutating func apply(completion: ((T)->T)) {
		self.data = data.map(completion)
	}
	
	func row(index: Int) -> [T] {
		var row = [T]()
		(0..<columns).forEach { (col) in
			row.append(self[index,col])
		}
		return row
	}
	
	func col(index: Int) -> [T] {
		var column = [T]()
		(0..<rows).forEach { (row) in
			column.append(self[row,index])
		}
		return column
	}
	
	subscript(row: Int, col: Int) -> T {
		get { return data[(row * columns) + col] }
		set { data[(row * columns) + col] = newValue }
	}
	
	func copy() -> Matrix {
		let cp = Matrix(data: data, rows: rows, columns: columns)
		return cp
	}
	
	static func +(left: Matrix, right: Matrix) -> Matrix {
		precondition(left.rows == right.rows && left.columns == right.columns)
		var resultMatrix = Matrix(data: left.data, rows: left.rows, columns: left.columns)
		for row in 0..<left.rows {
			for col in 0..<left.columns {
				resultMatrix[row,col] += right[row,col]
			}
		}
		return resultMatrix
	}
	
	static postfix func ^(matrix: Matrix) -> Matrix {
		var resultingMatrix = Matrix(rows: matrix.columns, columns: matrix.rows)
		for row in 0..<matrix.rows {
			for col in 0..<matrix.columns {
				resultingMatrix[col,row] = matrix[row,col]
			}
		}
		return resultingMatrix
	}
	
	static func *(left: Matrix, right: Matrix) -> Matrix {
		var leftCopy = left.copy()
		var rightCopy = right.copy()
		
		if (leftCopy.rows == 1 && rightCopy.rows == 1) && (leftCopy.columns == rightCopy.columns) {
			rightCopy = rightCopy^
		}
		else if (leftCopy.columns == 1 && rightCopy.columns == 1) && (leftCopy.rows == rightCopy.rows) {
			leftCopy = leftCopy^
		}
		precondition(leftCopy.columns == rightCopy.rows, "Matrices cannot be multipied")
		var dot = Matrix(rows: leftCopy.rows, columns: rightCopy.columns)
		
		for leftRow in 0..<leftCopy.rows {
			for rightColumn in 0..<rightCopy.columns {
				let a = vectorProduct(left: leftCopy.row(index: leftRow), right: rightCopy.col(index: rightColumn))
				dot[leftRow,rightColumn] = a
			}
		}
		return dot
	}
	
	private static func vectorProduct(left: [T], right: [T]) -> T {
		var d: T = 0
		for i in 0..<left.count {
			d += left[i] * right[i]
		}
		return d
	}
}

extension Matrix: Equatable {}

extension Matrix: CustomStringConvertible {
	var description: String {
		var description = ""
		for row in 0..<rows {
			for col in 0..<columns {
				description += "\(self[row,col])"
				if (col + 1) < columns {
					description += ", "
				}
			}
			description += "\n"
		}
		return description
	}
}


var matrix1: Matrix<Int> = Matrix(data: [1,2,3,4,5,6], rows: 2, columns: 3)
var matrix2: Matrix<Int> = Matrix(data: [7,8,9,10,11,12], rows: 3, columns: 2)

//print(matrix1 * matrix2)
//print(matrix1^)
//print(matrix1 + matrix1)


let e = Decimal(floatLiteral: 2.71828)
let one = Decimal(integerLiteral: 1)

extension Decimal {
	
	/// Returns a random decimal between 0.0 and 1.0, inclusive.
	public static var random: Decimal {
		return Decimal(arc4random()) / 0xFFFFFFFF
	}
	
	/// Random decimal between 0 and n-1.
	///
	/// - Parameter n:  Interval max
	/// - Returns:      Returns a random double point number between 0 and n max
	public static func random(min: Decimal, max: Decimal) -> Decimal {
		return Decimal.random * (max - min) + min
	}
}

func pow(base: Decimal,_ exponenet: Decimal) -> Decimal {
	return Decimal(pow(Double(truncating: base as NSNumber), Double(truncating: exponenet as NSNumber)))
}

protocol NormalizingFunctionType {
	func calculate(x: Decimal) -> Decimal
}

enum NormalizingFunction: NormalizingFunctionType {
	case sigmoid
	
	func calculate(x: Decimal) -> Decimal {
		switch self {
		case .sigmoid:
			var x = x
			x.negate()
			return one / (pow(base: e, x) + one)
		}
	}
}

protocol InitialSynapticWeightGeneratorType {
	func matrix(rows: Int, cols: Int) -> Matrix<Decimal>
}

enum InitialSynapticWeightGenerator: InitialSynapticWeightGeneratorType {
	case random
	
	func matrix(rows: Int, cols: Int) -> Matrix<Decimal> {
		let inputs = (0..<rows * cols).map { _ -> Decimal in
			return Decimal.random(min: 0, max: 1)
		}
		return Matrix<Decimal>(data: inputs, rows: rows, columns: cols)
	}
}

protocol ArtificalNeuralNetworkable {
	var normalizingFunction: NormalizingFunctionType { get }
	var initialSynapticWeightGenerator: InitialSynapticWeightGeneratorType { get }
	var synapticWeights: Matrix<Decimal> { get set }

	var trainingInputs: Matrix<Decimal> { get }
	var trainingOutputs: Matrix<Decimal> { get }
	
	func train(iterations: Int)
}

class ArtificalNeuralNetwork: ArtificalNeuralNetworkable {
	var normalizingFunction: NormalizingFunctionType {
		return NormalizingFunction.sigmoid
	}
	var initialSynapticWeightGenerator: InitialSynapticWeightGeneratorType {
		return InitialSynapticWeightGenerator.random
	}
	
	var trainingInputs: Matrix<Decimal>
	var trainingOutputs: Matrix<Decimal>
	var synapticWeights: Matrix<Decimal>
	
	init(inputs: Matrix<Decimal>, outputs: Matrix<Decimal>) {
		self.trainingInputs = inputs
		self.trainingOutputs = outputs
		self.synapticWeights = Matrix()
		self.synapticWeights = initialSynapticWeightGenerator.matrix(rows: trainingInputs.columns, cols: 1)
	}
	
	func train(iterations: Int) {
		(0..<iterations).forEach { (iterations) in
			print("Iteration: \(iterations)")
			let inputLayer = trainingInputs
			print(inputLayer)
			var result = inputLayer * synapticWeights
			print(result)
			result.apply(completion: { normalizingFunction.calculate(x: $0) })
			print(result)
			
			JSONEncoder().encode(<#T##value: Encodable##Encodable#>)
		}
	}
}

var trainingInputs: Matrix<Decimal> = Matrix(data: [0,0,1,1,1,1,1,0,1,0,1,1], rows: 4, columns: 3)
var trainingOutputs: Matrix<Decimal> = Matrix(data: [0,1,1,0], rows: 4, columns: 1)
ArtificalNeuralNetwork(inputs: trainingInputs, outputs: trainingOutputs).train(iterations: 3)
